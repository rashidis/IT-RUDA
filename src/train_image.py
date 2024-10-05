### rewriting every thing for deterministic KL, to make sure nothing is wrong
## change optimizer from GSD to Adam? no
## change cross entropy loss to nll? yes
## change embedding size to 16
import argparse
import os
import os.path as osp
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
from data_list import ImageList
import datetime
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm
import torch.nn.functional as F
import torch.distributions.independent as dist2
import torch.distributions as dist

from visualization import visualize



def image_classification_test(loader, model, iterrr):
    start_test = True
    with torch.no_grad():
 
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs = model(inputs)

            if config['augment_softmax'] != 0.0:
                K = 1 - config['augment_softmax'] * outputs .shape[1]
                outputs = outputs *K + config['augment_softmax']

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)

    num_categories = config["network"]["params"]["class_num"]

    if args.exp=='CS' or args.exp=='PDA':
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    elif args.exp=='OS':
        all_label_new=all_label[torch.where(all_label!=num_categories-1)]
        predict_new=predict[torch.where(all_label!=num_categories-1)]
        accuracy = torch.sum(torch.squeeze(predict_new).float() == all_label_new).item() / float(all_label_new.size()[0])

    #print('visualizing')
    #visualize(config,loader,model,iterrr,accuracy )

    return accuracy


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=0, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=0, drop_last=True)
    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                          shuffle=False, num_workers=0)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    parameter_list = base_network.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))
    print(optimizer)
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = nn.DataParallel(base_network)

    ## train  
    print('train started') 
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    print('number of source and target batches:',len_train_source, len_train_target)
    best_acc = 0.0
    best_model = nn.Sequential(base_network)
    each_log = ""
    for i in range(config["num_iterations"]):
        if i==0:
            temp_acc = image_classification_test(dset_loaders, base_network, i)
        if i % config["test_interval"] == config["test_interval"]-1:
            print("testing")
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, base_network, i)
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:   #save model is it is better than before
                best_acc = temp_acc
                best_model = temp_model
                torch.save(best_model, config['model_output_path'] + "{}_{}".
                         format(config['log_name'], str(best_acc)) +config['figure_name']  )


            if args.dmetric=='kl' and args.kl_reg_aux !=0:
                log_str = "iter: {:05d}, test acc.: {:.5f}, cls_loss:{:.4f},kl_loss:{:.4f}, kl_aux_loss:{:.4f}, total_loss:{:.4f}" \
                .format(i, temp_acc, cls_loss.item(), kl.item(), kl_aux.item(),obj.item())
            if args.dmetric=='kl' and args.kl_reg_aux ==0:
                log_str = "iter: {:05d}, test acc.: {:.5f}, cls_loss:{:.4f}, kl_loss:{:.4f}, total_loss:{:.4f}" \
                .format(i, temp_acc, cls_loss.item(), kl.item(),obj.item())
            elif args.dmetric=='alpha':
                log_str = "iter: {:05d}, test acc.: {:.5f}, cls_loss:{:.4f}, alpha_loss:{:.4f}, total_loss:{:.4f}" \
                .format(i, temp_acc, cls_loss.item(), alpha_loss.item(),obj.item())
            elif 'init' in args.dmetric:
                log_str = "iter: {:05d}, test acc.: {:.5f}, cls_loss:{:.4f}, kl_loss:{:.4f}, kl_aux_loss:{:.4f}, total_loss:{:.4f}" \
                .format(i, temp_acc, cls_loss.item(), kl.item(),kl_aux.item(),obj.item())
            elif args.dmetric=='source':
                log_str = "iter: {:05d}, test acc.: {:.5f}, cls_loss:{:.4f}, total_loss:{:.4f}" \
                .format(i, temp_acc, cls_loss.item(),obj.item()) 
            print(log_str)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            

            config["out_file"].write(each_log)
            config["out_file"].flush()
            each_log = ""

        ################ train one iter #######################
        base_network.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        ### load data
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        
        ### get features
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        
        #### calc clss_loss
        #outputs = torch.cat((outputs_source, outputs_target), dim=0)
        #softmax_out = nn.Softmax(dim=1)(outputs) 
        #entropy = loss.Entropy(softmax_out)
        if config['augment_softmax'] != 0.0:
            K = 1 - config['augment_softmax'] * outputs_source.shape[1]
            outputs_source= outputs_source*K + config['augment_softmax']
        cls_loss  = F.nll_loss(F.log_softmax(outputs_source),labels_source)
        #classifier_loss = nn.CrossEntropyLoss()(outputs_source,labels_source)

        ### create distibutions
        total_z_sigma = config['sigma_coef']* torch.ones(features.shape)
        total_z_sigma=total_z_sigma.cuda()
        z_mu, z_sigma = features[:inputs_source.shape[0]], total_z_sigma[:inputs_source.shape[0]]
        z_mu_target, z_sigma_target = features[inputs_source.shape[0]:], total_z_sigma[inputs_source.shape[0]:]
        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
        z_dist_target = dist.Independent(dist.normal.Normal(z_mu_target,z_sigma_target),1)

        mix_coeff = dist.categorical.Categorical(inputs_source.new_ones(inputs_source.shape[0]))
        mixture = dist.mixture_same_family.MixtureSameFamily(mix_coeff,z_dist)
        mix_coeff_target = dist.categorical.Categorical(inputs_target.new_ones(inputs_target.shape[0]))
        mixture_target = dist.mixture_same_family.MixtureSameFamily(mix_coeff_target,z_dist_target)

        ### alpha or KL loss
        obj =args.cls_weight * cls_loss
        if args.kl_reg != 0.0:
            kl = cls_loss.new_zeros([])
            if i==0:
                print('kl')
            kl = (mixture_target.log_prob(z_mu_target)-mixture.log_prob(z_mu_target)).mean()
            obj = obj + args.kl_reg*kl
        if args.kl_reg_aux != 0.0:
            kl_aux = cls_loss.new_zeros([])
            if i==0:
                print('kl_aux')
            kl_aux = (mixture.log_prob(z_mu)-mixture_target.log_prob(z_mu)).mean()
            obj = obj + args.kl_reg_aux*kl_aux

        if args.alpha_reg !=0.0:
            alpha_loss = cls_loss.new_zeros([])
            if i==0:
                print('alpha')
            alpha=args.alpha
            if args.exp=='PDA':
                if i==0:
                    print('special PDA')
                alpha_loss=  ( torch.exp( (1-alpha)*(mixture_target.log_prob(z_mu)-mixture.log_prob(z_mu)) )-1).mean()/(alpha*(alpha-1))
            else:
                alpha_loss=  ( torch.exp( (1-alpha)*(mixture.log_prob(z_mu_target)-mixture_target.log_prob(z_mu_target)) )-1).mean()/(alpha*(alpha-1))
            obj = obj + args.alpha_reg*alpha_loss

        obj.backward()
        optimizer.step()

        ### print log
        if args.dmetric=='kl' and args.kl_reg_aux!=0:
            log_str = "iter: {:05d},cls_loss:{:.4f}, kl_loss:{:.4f}, kl_aux_loss:{:.4f},total_loss:{:.4f}" \
            .format(i, cls_loss.item(), kl.item(),kl_aux.item(),obj.item())
        if args.dmetric=='kl' and args.kl_reg_aux==0:
            log_str = "iter: {:05d},cls_loss:{:.4f}, kl_loss:{:.4f},total_loss:{:.4f}" \
            .format(i, cls_loss.item(), kl.item(),obj.item())
        elif args.dmetric=='alpha':
            log_str = "iter: {:05d},cls_loss:{:.4f}, alpha_loss:{:.4f},total_loss:{:.4f}" \
            .format(i, cls_loss.item(), alpha_loss.item(),obj.item())
        elif 'init' in args.dmetric:
            log_str = "iter: {:05d},cls_loss:{:.4f}, kl_loss:{:.4f},kl_aux:{:4f},total_loss:{:.4f}" \
            .format(i, cls_loss.item(), kl.item(), kl_aux.item(),obj.item())
        elif args.dmetric=='source':
            log_str = "iter: {:05d}, cls_loss:{:.4f}, total_loss:{:.4f}" \
            .format(i, cls_loss.item(),obj.item())
        each_log += log_str + "\n"
        #print(log_str)


    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50',
                        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13",
                                 "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])


    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument("--use_seed", type=bool, default=False)
    parser.add_argument("--torch_seed", type=int, default=1)
    parser.add_argument("--torch_cuda_seed", type=int, default=1)

    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument("--num_classes", type=int, default=31) #31 for office, 65 for office-home
    parser.add_argument("--num_shared", type=int, default=10) #31 for office, 65 for office-home

    parser.add_argument("--epoch", type=int, default=40000)

    parser.add_argument("--cls_weight", type=float, default=1)
    parser.add_argument("--kl_reg", type=float, default=0)
    parser.add_argument("--kl_reg_aux", type=float, default=0)

    parser.add_argument("--alpha_reg", type=float, default=0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--sigma", type=float, default=1)
    parser.add_argument("--outlier_num", type=int, default=21)

    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--dmetric", type=str, default='alpha')
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'],
                        help="The dataset or source dataset used")
    parser.add_argument("--exp", type=str, default="cs")
    parser.add_argument("--log_name", type=str, default="a2w")
    parser.add_argument('--s_dset_path', type=str, default='data/office/amazon_list.txt',
                        help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='data/office/webcam_list.txt',
                        help="The target dataset path list")


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if (args.use_seed):
        torch.manual_seed(args.torch_seed)
        torch.cuda.manual_seed(args.torch_cuda_seed)
        torch.cuda.manual_seed_all(args.torch_cuda_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    config = {}
#========================================================
    config['sigma_coef']=args.sigma #0.001
    config['augment_softmax']=0.05
#========================================================
    config["gpu"] = args.gpu_id
    config['torch_seed'] = torch.initial_seed()
    # config['torch_cuda_seed'] = torch.cuda.initial_seed()
    config["num_iterations"] = args.epoch
    config["test_interval"] = args.test_interval

    ### set path for the outputs
    params_name='_out'+str(args.outlier_num)+'_lr'+str(args.lr)+'_bs'+str(args.bs)+'_sigma'+str(args.sigma)+'_clsW'+str(args.cls_weight)+'_'
    print(params_name)
    if args.alpha_reg!=0:
        params_name=params_name+'alphaW'+str(args.alpha_reg)+'_alpha'+str(args.alpha)+'_'
    if args.kl_reg!=0:
        params_name=params_name+'klW'+str(args.kl_reg)+'_'
    if args.kl_reg_aux!=0:
        params_name=params_name+'auxW'+str(args.kl_reg_aux)+'_'

    config['figure_name']=args.log_name+params_name+args.dmetric+'_'+args.exp
    print(config['figure_name'])
    if not osp.exists("logs_" + args.dset +"_"+args.exp+ "/" + args.log_name + "/"+config['figure_name']+'/'):
        os.system('mkdir -p ' + "logs_" + args.dset +"_"+args.exp+ "/" + args.log_name + "/"+config['figure_name']+'/')  
    config["log_output_path"] = "logs_" + args.dset +"_"+args.exp+ "/" + args.log_name + '/'+config['figure_name']+"/log/"
    config["model_output_path"] = "logs_" + args.dset +"_"+args.exp+ "/" + args.log_name + '/'+config['figure_name']+"/model/"
    config['log_name'] = args.log_name
    if not osp.exists(config["log_output_path"]):
        os.system('mkdir -p ' + config["log_output_path"])
    config["out_file"] = open(
        osp.join(config["log_output_path"], args.log_name + "_{}.txt".format(str(datetime.datetime.utcnow()))), "w")
    if not osp.exists(config["log_output_path"]):
        os.mkdir(config["log_output_path"])
    if not osp.exists(config["model_output_path"]):
        os.mkdir(config["model_output_path"])


    config["prep"] = {'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name": network.AlexNetFc, \
                             "params": {"use_bottleneck": True, "bottleneck_dim": args.dim, "new_cls": True}}
    elif "ResNet" in args.net:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": True, "bottleneck_dim": args.dim,
                                        "new_cls": True}}
    elif "VGG" in args.net:
        config["network"] = {"name": network.VGGFc, \
                             "params": {"vgg_name": args.net, "use_bottleneck": True, "bottleneck_dim": args.dim,
                                        "new_cls": True}}

    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9, \
                                                               "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv", \
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}

    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": args.bs}, \
                      "target": {"list_path": args.t_dset_path, "batch_size": args.bs}, \
                      "test": {"list_path": args.t_dset_path, "batch_size": args.bs}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters 0.001 default
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters 0.0003 default
        config["network"]["params"]["class_num"] = args.num_classes# 31
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = args.num_classes #12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = args.num_classes #12
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = args.num_classes  #65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')


    config["out_file"].write(str(config) + "\n")
    config["out_file"].flush()
    print('args',args)
    print('config',config)
    train(config)
