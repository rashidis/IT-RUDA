def get_dat_for_vis(loaderr, model):
    """generate daya for visualization"""
    test_predictions = []
    test_labels = []
    test_embeddings = torch.zeros((0, args.dim), dtype=torch.float32)
    iter_sample = loaderr.__iter__()
    for i in range(len(loaderr)):
        data = iter_sample.next()
        inputs = data[0]
        labels = data[1]
        inputs = inputs.cuda()
        embeddings, outputs = model(inputs)
################################
        if config['augment_softmax'] != 0.0:
            K = 1 - config['augment_softmax'] * outputs .shape[1]
            outputs = outputs *K + config['augment_softmax']
################################
        _, predicts = torch.max(outputs, 1)
        test_predictions.extend(predicts.detach().cpu().tolist())
        test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), 0)
        test_labels.extend(labels.detach().cpu().tolist())
    return(np.array(test_embeddings),np.array(test_predictions),np.array(test_labels))

def visualize(config,loaderr,model,iterrr,acc):
    if iterrr==0:
        print('try visualization in the first iteration')
   
    num_categories = config["network"]["params"]["class_num"]

    test_embeddings, test_predictions,test_labels=get_dat_for_vis(loaderr['test'],model)
    source_embeddings, source_predictions,source_labels=get_dat_for_vis(loaderr['source'],model)
    source_accuracy = np.sum(source_predictions == source_labels) / float(np.size(source_labels))
    print('source_accuracy',source_accuracy)
    
    all_embeddings=np.concatenate((test_embeddings,source_embeddings),axis=0)
    
    tsne_proj_all = TSNE(2).fit_transform(all_embeddings)
    tsne_proj_test=tsne_proj_all[0:np.shape(test_embeddings)[0]]
    tsne_proj_source=tsne_proj_all[np.shape(test_embeddings)[0]:np.shape(source_embeddings)[0]+np.shape(test_embeddings)[0]]


    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = matplotlib.cm.get_cmap('tab20')
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3,figsize=(18,5))
    if args.exp=='OS':
        tsne_proj_test_no10=tsne_proj_test[np.where(test_labels!=num_categories-1)]
        test_predictions_no10=test_predictions[np.where(test_labels!=num_categories-1)]
        test_labels_no10=test_labels[np.where(test_labels!=num_categories-1)]
        list_labels=source_labels
    elif args.exp=='PDA' or args.exp=='CS':
        tsne_proj_test_no10=tsne_proj_test
        test_predictions_no10=test_predictions#[np.where(test_labels<args.num_shared)]
        test_labels_no10=test_labels#[np.where(test_labels<args.num_shared)]
        list_labels=test_labels

    private_list=[x for x in list(np.arange(num_categories)) if x not in list(np.unique(list_labels))]
    shared_list=list(np.unique(list_labels))
   # print('testttt',shared_list)
    #print('testttt private',private_list)

    for lab in shared_list:
        indices_test = np.where(test_predictions_no10==lab)
        ax1.scatter(tsne_proj_test_no10[indices_test,0],tsne_proj_test_no10[indices_test,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
        indices_target = np.where(test_labels_no10==lab)
        ax2.scatter(tsne_proj_test_no10[indices_target,0],tsne_proj_test_no10[indices_target,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
        indices_source =  np.where(source_predictions==lab)
        ax3.scatter(tsne_proj_source[indices_source,0],tsne_proj_source[indices_source,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
    if args.exp=='PDA':
        for lab in private_list: 
            indices_source =  np.where(source_predictions==lab)
            ax3.scatter(tsne_proj_source[indices_source,0],tsne_proj_source[indices_source,1], c='k', marker='+', label ="other" if lab == private_list[0] else "" ,alpha=0.1)
            indices_test = np.where(test_predictions_no10==lab)
            ax1.scatter(tsne_proj_test_no10[indices_test,0],tsne_proj_test_no10[indices_test,1],c='k', marker='+', label ="other" if lab == private_list[0] else "" ,alpha=0.1)
    elif args.exp=='OS':
        for lab in private_list:
            print(lab)
            indices_target = np.where(test_labels==lab)#ground truth
            ax2.scatter(tsne_proj_test[indices_target,0],tsne_proj_test[indices_target,1],c='k', marker='+', label ="other" if lab == private_list[0] else "" ,alpha=0.1)

    #ax1.legend(fontsize='large', markerscale=2)
    #ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_title('test_pred'+'_acc. {:.4f}'.format(acc))
    ax1.set_xlim(-70,70)
    ax1.set_ylim(-70,70)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_title('test_labels'+'_iter'+str(iterrr))
    ax2.set_xlim(-70,70)
    ax2.set_ylim(-70,70)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.set_title('source'+'_acc. {:.4f}'.format(source_accuracy))
    ax3.set_xlim(-70,70)
    ax3.set_ylim(-70,70)
    plt.savefig("logs_" + args.dset +"_"+args.exp+ "/" + args.log_name + "/"+config['figure_name']+'/'+config['log_name']+'_iter'+str(iterrr)+'.png')
    
    fig=plt.figure()
    for lab in shared_list:
        indices_source =  np.where(source_predictions==lab)
        plt.scatter(tsne_proj_source[indices_source,0],tsne_proj_source[indices_source,1], c='b')
        indices_test = np.where(test_predictions_no10==lab)
        plt.scatter(tsne_proj_test_no10[indices_test,0],tsne_proj_test_no10[indices_test,1], c='r')

    if args.exp=='PDA':
        for lab in private_list: 
            indices_source =  np.where(source_predictions==lab)
            plt.scatter(tsne_proj_source[indices_source,0],tsne_proj_source[indices_source,1], c='b')
            indices_test = np.where(test_predictions_no10==lab)
            plt.scatter(tsne_proj_test_no10[indices_test,0],tsne_proj_test_no10[indices_test,1],c='r')
    elif args.exp=='OS':
        for lab in private_list:
            print(lab)
            indices_target = np.where(test_labels==lab)#ground truth
            plt.scatter(tsne_proj_test[indices_target,0],tsne_proj_test[indices_target,1],c='r')
    plt.axis('off')
    plt.savefig("logs_"+ args.dset +"_"+args.exp+ "/" + args.log_name + "/"+config['figure_name']+'/'+"paper_" +config['log_name']+'_iter'+str(iterrr)+'.png')
