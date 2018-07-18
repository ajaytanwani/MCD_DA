from __future__ import division

import json
import numpy as np
import os

import torch
import tqdm
from PIL import Image
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
from argmyparse import add_additional_params_to_args, fix_img_shape_args, get_da_mcd_training_parser
from datasets import ConcatDataset, get_dataset, check_src_tgt_ok
from loss import CrossEntropyLoss2d, get_prob_distance_criterion
from eval import label_mapping
from models.model_util import get_models, get_optimizer
from transform import ReLabel, ToLabel, Scale, RandomSizedCrop, RandomHorizontalFlip, RandomRotation
from util import mkdir_if_not_exist, save_dic_to_json, check_if_done, save_checkpoint, adjust_learning_rate, \
    get_class_weight_from_file

# import ipdb; ipdb.set_trace()
parser = get_da_mcd_training_parser()
args_read = parser.parse_args()
args_read = add_additional_params_to_args(args_read)
args_read = fix_img_shape_args(args_read)
check_src_tgt_ok(args_read.src_dataset, args_read.tgt_dataset)
# import ipdb; ipdb.set_trace()
weight = torch.ones(args_read.n_class)
# if not args_read.add_bg_loss:
weight[args_read.n_class - 1] = 0  # Ignore background loss

args_read.start_epoch = 0
resume_flg = True if args_read.resume else False
start_epoch = 0
if args_read.resume:
    print("=> loading checkpoint '{}'".format(args_read.resume))
    if not os.path.exists(args_read.resume):
        raise OSError("%s does not exist!" % args_read.resume)

    indir, infn = os.path.split(args_read.resume)

    old_savename = args_read.savename
    args_read.savename = infn.split("-")[0]
    print ("savename is %s (original savename %s was overwritten)" % (args_read.savename, old_savename))

    checkpoint = torch.load(args_read.resume)
    start_epoch = checkpoint["epoch"]
    # ---------- Replace Args!!! ----------- #
    args_read = checkpoint['args']
    # -------------------------------------- #
    model_g, model_f1, model_f2 = get_models(net_name=args_read.net, res=args_read.res, input_ch=args_read.input_ch,
                                             n_class=args_read.n_class, method=args_read.method,
                                             is_data_parallel=args_read.is_data_parallel)
    optimizer_g = get_optimizer(model_g.parameters(), lr=args_read.lr, momentum=args_read.momentum, opt=args_read.opt,
                                weight_decay=args_read.weight_decay)
    optimizer_f = get_optimizer(list(model_f1.parameters()) + list(model_f2.parameters()), lr=args_read.lr, opt=args_read.opt,
                                momentum=args_read.momentum, weight_decay=args_read.weight_decay)

    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_f1.load_state_dict(checkpoint['f1_state_dict'])
    if not args_read.uses_one_classifier:
        model_f2.load_state_dict(checkpoint['f2_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    optimizer_f.load_state_dict(checkpoint['optimizer_f'])
    print("=> loaded checkpoint '{}'".format(args_read.resume))

else:
    model_g, model_f1, model_f2 = get_models(net_name=args_read.net, res=args_read.res, input_ch=args_read.input_ch,
                                             n_class=args_read.n_class,
                                             method=args_read.method, uses_one_classifier=args_read.uses_one_classifier,
                                             is_data_parallel=args_read.is_data_parallel)
    optimizer_g = get_optimizer(model_g.parameters(), lr=args_read.lr, momentum=args_read.momentum, opt=args_read.opt,
                                weight_decay=args_read.weight_decay)
    optimizer_f = get_optimizer(list(model_f1.parameters()) + list(model_f2.parameters()), opt=args_read.opt,
                                lr=args_read.lr, momentum=args_read.momentum, weight_decay=args_read.weight_decay)
if args_read.uses_one_classifier:
    print ("f1 and f2 are same!")
    model_f2 = model_f1

mode = "%s-%s2%s-%s_%sch" % (args_read.src_dataset, args_read.src_split, args_read.tgt_dataset, args_read.tgt_split, args_read.input_ch)
if args_read.net in ["fcn", "psp"]:
    model_name = "%s-%s-%s-res%s" % (args_read.method, args_read.savename, args_read.net, args_read.res)
else:
    model_name = "%s-%s-%s" % (args_read.method, args_read.savename, args_read.net)

outdir = os.path.join(args_read.base_outdir, mode)

# import ipdb; ipdb.set_trace()
# Create Model Dir
pth_dir = os.path.join(outdir, "pth")
mkdir_if_not_exist(pth_dir)

# Create Model Dir and  Set TF-Logger
tflog_dir = os.path.join(outdir, "tflog", model_name)
mkdir_if_not_exist(tflog_dir)
configure(tflog_dir, flush_secs=5)

# Save param dic
if resume_flg:
    json_fn = os.path.join(args_read.outdir, "param-%s_resume.json" % model_name)
else:
    json_fn = os.path.join(outdir, "param-%s.json" % model_name)
# check_if_done(json_fn)
save_dic_to_json(args_read.__dict__, json_fn)

train_img_shape = tuple([int(x) for x in args_read.train_img_shape])
img_transform_list = [
    Scale(train_img_shape, Image.BILINEAR),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225])
]
if args_read.augment:
    aug_list = [
        RandomRotation(),
        RandomHorizontalFlip(),
        RandomSizedCrop()
    ]
    img_transform_list = aug_list + img_transform_list

img_transform = Compose(img_transform_list)

label_transform = Compose([
    Scale(train_img_shape, Image.NEAREST),
    ToLabel(),
    ReLabel(255, args_read.n_class - 1),  # Last Class is "Void" or "Background" class
])

if args_read.src_dataset == "gta":
    info_json_fn = "./dataset/city_info.json"
    with open(info_json_fn) as f:
        info = json.load(f)
        mapping = np.array(info['label2train'], dtype=np.int)

src_dataset = get_dataset(dataset_name=args_read.src_dataset, split=args_read.src_split, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=args_read.input_ch)

tgt_dataset = get_dataset(dataset_name=args_read.tgt_dataset, split=args_read.tgt_split, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=args_read.input_ch)

train_loader = torch.utils.data.DataLoader(
    ConcatDataset(
        src_dataset,
        tgt_dataset
    ),
    batch_size=args_read.batch_size, shuffle=True,
    pin_memory=True)

# weight = get_class_weight_from_file(n_class=args_read.n_class, weight_filename=args_read.loss_weights_file,
#                                     add_bg_loss=args_read.add_bg_loss)

weight = get_class_weight_from_file(n_class=args_read.n_class)

if torch.cuda.is_available():
    model_g.cuda()
    model_f1.cuda()
    model_f2.cuda()
    weight = weight.cuda()

criterion = CrossEntropyLoss2d(weight)
criterion_d = get_prob_distance_criterion(args_read.d_loss)

model_g.train()
model_f1.train()
model_f2.train()

for epoch in range(start_epoch, args_read.epochs):
    d_loss_per_epoch = 0
    c_loss_per_epoch = 0

    for ind, (source, target) in tqdm.tqdm(enumerate(train_loader)):
        src_imgs, src_lbls = Variable(source[0]), Variable(source[1])
        tgt_imgs = Variable(target[0])


        print torch.max(src_lbls)
        # import ipdb;
        #
        # ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        # src_lbls = np.expand_dims(label_mapping(np.squeeze(src_lbls.data.numpy()), mapping), 0)
        # src_lbls = torch.tensor(src_lbls)

        if torch.cuda.is_available():
            src_imgs, src_lbls, tgt_imgs = src_imgs.cuda(), src_lbls.cuda(), tgt_imgs.cuda()

        # update generator and classifiers by source samples
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        loss = 0
        loss_weight = [1.0, 1.0]
        outputs = model_g(src_imgs)

        outputs1 = model_f1(outputs)
        outputs2 = model_f2(outputs)

        loss += criterion(outputs1, src_lbls)
        loss += criterion(outputs2, src_lbls)
        loss.backward()
        c_loss = loss.data[0]
        c_loss_per_epoch += c_loss

        optimizer_g.step()
        optimizer_f.step()
        # update for classifiers
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        outputs = model_g(src_imgs)
        outputs1 = model_f1(outputs)
        outputs2 = model_f2(outputs)
        loss = 0
        loss += criterion(outputs1, src_lbls)
        loss += criterion(outputs2, src_lbls)
        outputs = model_g(tgt_imgs)
        outputs1 = model_f1(outputs)
        outputs2 = model_f2(outputs)
        loss -= criterion_d(outputs1, outputs2)
        loss.backward()
        optimizer_f.step()

        d_loss = 0.0
        # update generator by discrepancy
        for i in xrange(args_read.num_k):
            optimizer_g.zero_grad()
            loss = 0
            outputs = model_g(tgt_imgs)
            outputs1 = model_f1(outputs)
            outputs2 = model_f2(outputs)
            loss += criterion_d(outputs1, outputs2) * args_read.num_multiply_d_loss
            loss.backward()
            optimizer_g.step()

        d_loss += loss.data[0] / args_read.num_k
        d_loss_per_epoch += d_loss
        if ind % 100 == 0:
            print("iter [%d] DLoss: %.6f CLoss: %.4f" % (ind, d_loss, c_loss))

        if ind > args_read.max_iter:
            break

    print("Epoch [%d] DLoss: %.4f CLoss: %.4f" % (epoch, d_loss_per_epoch, c_loss_per_epoch))

    log_value('c_loss', c_loss_per_epoch, epoch)
    log_value('d_loss', d_loss_per_epoch, epoch)
    log_value('lr', args_read.lr, epoch)

    if args_read.adjust_lr:
        args_read.lr = adjust_learning_rate(optimizer_g, args_read.lr, args_read.weight_decay, epoch, args_read.epochs)
        args_read.lr = adjust_learning_rate(optimizer_f, args_read.lr, args_read.weight_decay, epoch, args_read.epochs)

    checkpoint_fn = os.path.join(pth_dir, "%s-%s.pth.tar" % (model_name, epoch + 1))
    args_read.start_epoch = epoch + 1
    save_dic = {
        'epoch': epoch + 1,
        'args': args_read,
        'g_state_dict': model_g.state_dict(),
        'f1_state_dict': model_f1.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_f': optimizer_f.state_dict(),
    }
    if not args_read.uses_one_classifier:
        save_dic['f2_state_dict'] = model_f2.state_dict()

    save_checkpoint(save_dic, is_best=False, filename=checkpoint_fn)
