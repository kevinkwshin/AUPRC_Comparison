import sklearn
import scipy.stats

def pValue(mean_ref,std_ref,mean_target, std_target,df):
    t = (mean_ref-mean_target)/np.sqrt((std_ref**2)/df +(std_target**2)/df)
    return scipy.stats.t.sf(abs(t), df=df-1)*2

def AUPRC_comparison(y_ref, yhat_ref, yhat_target, bootstrap=10):
    
    assert len(y_ref)==len(yhat_ref) == len(yhat_target)
    yhat_ref_means =[]
    yhat_target_means =[]
    
    for i in range(bootstrap):
        sample = sklearn.model_selection.train_test_split(y_ref, yhat_ref, yhat_target, train_size=0.8, random_state=None)
        y_ref_sample, _, yhat_ref_sample, _, yhat_target_sample,_ = sample
        
        auprc_ref = sklearn.metrics.average_precision_score(y_ref_sample,yhat_ref_sample)
        auprc_target = sklearn.metrics.average_precision_score(y_ref_sample,yhat_target_sample)
        
        yhat_ref_means.append(auprc_ref)
        yhat_target_means.append(auprc_target)
        
    mean_ref = np.mean(yhat_ref_means)
    std_ref = np.std(yhat_ref_means)
    
    mean_target = np.mean(yhat_target_means)
    std_target = np.std(yhat_target_means)
    pvalue = pValue(mean_ref, std_ref, mean_target, std_target, bootstrap)
     
    return {'auprc_ref':auprc_ref, 'auprc_target':auprc_target, 'pvalue':pvalue}

a = np.array([0,0,0,0,0,1,1,1,1,1])
b = np.array([0.3,0.2,0.4,0.5,0.6,.7,.6,.8,.3,.4])
c = np.array([0.1,0.1,0.1,0.5,0.4,.7,.6,.8,.3,.4])

AUPRC_comparison(a,b,c)
