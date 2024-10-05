import numpy as np
import torch
import torch.nn as nn
import scipy.special
import torch.nn.functional as F


def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def CDAN(input_list, args, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda(device=int(args.gpu_id))
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0

        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        l = nn.BCELoss(reduction='none')(ad_out, dc_target)
        return torch.sum(weight.view(-1, 1) * nn.BCELoss()(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)


def mdd_loss(features, labels, left_weight=1, right_weight=1):
    softmax_out = nn.Softmax(dim=1)(features)
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        raise Exception('Incorrect batch size provided')

    batch_left = softmax_out[:int(0.5 * batch_size)]
    batch_right = softmax_out[int(0.5 * batch_size):]

    loss = torch.norm((batch_left - batch_right).abs(), 2, 1).sum() / float(batch_size)

    labels_left = labels[:int(0.5 * batch_size)]
    batch_left_loss = get_pari_loss1(labels_left, batch_left)

    labels_right = labels[int(0.5 * batch_size):]
    batch_right_loss = get_pari_loss1(labels_right, batch_right)
    return loss + left_weight * batch_left_loss + right_weight * batch_right_loss



def alpha_ours(features_source, features_target,alpha,bw):
    n=features_target.size()[0]
    m=features_source.size()[0]
    res=0
    for target in features_target:
        val=0
        for source in features_source:
            val=val+torch.exp(  -1*torch.sqrt(torch.linalg.norm(target-source,ord=2))/(bw**2)   )
    res=torch.pow(val,1-alpha)+res
    return(-1*res+n*m**(1-alpha))


def get_dist_knn(x,y,k):
    knn_dist=[]
    for i in range(0,x.size()[0]): #get the distance of x samples from y
        dist = torch.norm(x[i] - y, dim=1, p=None)    ##### or x-y[i]
        knn = dist.topk(k, largest=False)
        knn_dist.append(knn.values[-1])
        #print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))
    return(knn_dist)

def alpha(source,target,alpha,k):
    B=scipy.special.gamma(k)**2/(scipy.special.gamma(k-alpha+1)*scipy.special.gamma(k+alpha-1))
    #print('B',B)
    d=target.size()[1]
    #weights = torch.ones(target.size()[1]).expand(1, -1)
    #samples=torch.multinomial(weights, num_samples=d, replacement=False)[0]
    #source=source[:,samples]
    #target=target[:,samples]

    ro=torch.stack(get_dist_knn(source,source,k+1))
    vi=torch.stack(get_dist_knn(source,target,k))
    #print(ro)
    #print(vi)
    #print('ro/vi',ro/vi)
    ro_vi=torch.pow((ro/vi),d*(1-alpha))
    #print('ro_vi',ro_vi)
    #print('fraction',torch.sum(ro_vi))
    #print(d,d*(1-alpha))

    n=target.size()[0]
    m=source.size()[0]
    alll=torch.sum(ro_vi*((n-1)/m)**(1-alpha ))
    #print('n',n)
    #print('m',m)
    #print(((n-1)/m)**(1-alpha))
    #print('all')
    #print('alpha', (1-(B/n)*alll) / (alpha*(1-alpha)) )

    return( (1-(B/n)*alll) / (alpha*(1-alpha)) )




def OSNN_loss(loader_s,config):
    thr=config['knn_unk_thr']
    c=config["network"]["params"]["class_num"]


    iter_sample = loader_s.__iter__()
    for i in range(len(loader_s)):
        data = iter_sample.next()
        inputs = data[0]
        labels = data[1]
        inputs = inputs.cuda()
        print('input size',inputs.size())
        embeddings, outputs = model(inputs)
        _, predicts = torch.max(outputs, 1)


    cls=[]
    #Compute the distance of target samples from source samples
    dist = torch.norm(target [i] - source, dim=1, p=None)
    knn = dist.topk(2, largest=False)
        #print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))

        #if two closet samples have same label
    if Ys[knn.indices[0]]==Ys[knn.indices[1]]:
        cls.append(Ys[knn.indices[0]])
    else:
        # Compare ratio of distances with threshold T
        if knn.values[0]<=thr*knn.values[1]:
            cls.append(Ys[knn.indices[0]])
        elif knn.values[1]<=thr*knn.values[0]:
            cls.append(Ys[knn.indices[1]])
        else:
            cls.append(torch.tensor([c-1]))
    cls=torch.stack(cls)

 
def mdd_digit(features, labels, left_weight=1, right_weight=1, weight=1):
    softmax_out = nn.Softmax(dim=1)(features)
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        raise Exception('Incorrect batch size provided')

    batch_left = softmax_out[:int(0.5 * batch_size)]
    batch_right = softmax_out[int(0.5 * batch_size):]

    loss = torch.norm((batch_left - batch_right).abs(), 2, 1).sum() / float(batch_size)

    labels_left = labels[:int(0.5 * batch_size)]
    labels_left_left = labels_left[:int(0.25 * batch_size)]
    labels_left_right = labels_left[int(0.25 * batch_size):]

    batch_left_left = batch_left[:int(0.25 * batch_size)]
    batch_left_right = batch_left[int(0.25 * batch_size):]
    batch_left_loss = get_pair_loss(labels_left_left, labels_left_right, batch_left_left, batch_left_right)

    labels_right = labels[int(0.5 * batch_size):]
    labels_right_left = labels_right[:int(0.25 * batch_size)]
    labels_right_right = labels_right[int(0.25 * batch_size):]

    batch_right_left = batch_right[:int(0.25 * batch_size)]
    batch_right_right = batch_right[int(0.25 * batch_size):]
    batch_right_loss = get_pair_loss(labels_right_left, labels_right_right, batch_right_left, batch_right_right)

    return weight*loss + left_weight * batch_left_loss + right_weight * batch_right_loss


def get_pair_loss(labels_left, labels_right, features_left, features_right):
    loss = 0
    for i in range(len(labels_left)):
        if (labels_left[i] == labels_right[i]):
            loss += torch.norm((features_left[i] - features_right[i]).abs(), 2, 0).sum()
    return loss


def get_pari_loss1(labels, features):
    loss = 0
    count = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if (labels[i] == labels[j]):
                count += 1
                loss += torch.norm((features[i] - features[j]).abs(), 2, 0).sum()
    return loss / count


def EntropicConfusion(features):
    softmax_out = nn.Softmax(dim=1)(features)
    batch_size = features.size(0)
    loss = torch.mul(softmax_out, torch.log(softmax_out)).sum() * (1.0 / batch_size)
    return loss


def marginloss(yHat, y, classes, alpha, weight):
    batch_size = len(y)
    classes = classes
    yHat = F.softmax(yHat, dim=1)
    Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))#.detach()
    Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
    Px = yHat / Yg_.view(len(yHat), 1)
    Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
    y_zerohot = torch.ones(batch_size, classes).scatter_(
        1, y.view(batch_size, 1).data.cpu(), 0)

    output = Px * Px_log * y_zerohot.cuda()
    loss = torch.sum(output, dim=1)/ np.log(classes - 1)
    Yg_ = Yg_ ** alpha
    print(weight )
    if weight is not None:
        weight *= (Yg_.view(len(yHat), )/ Yg_.sum())
    else:
        weight = (Yg_.view(len(yHat), )/ Yg_.sum())

    weight = weight.detach()
    loss = torch.sum(weight * loss) / torch.sum(weight)

    return loss
