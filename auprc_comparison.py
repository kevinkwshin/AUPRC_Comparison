
import sklearn
import scipy.stats

def pValue(mean_ref,std_ref,mean_target, std_target,df):
    t = (mean_ref-mean_target)/np.sqrt((std_ref**2)/df +(std_target**2)/df)
    return scipy.stats.t.sf(abs(t), df=df-1)*2

def AUPRC_comparison(y_ref, yhat_ref, yhat_target, bootstrap=10):
    """
    y_ref : ground truth consisted with 0, 1
    yhat_ref : likelihood value of reference model consisted with 0 ~ 1
    yhat_target : likelihood value of target model consisted with 0 ~ 1

    example:
    a = np.array([0,0,0,0,0,0,1,1,1,1,1,1,1]) # GT
    b = np.array([0.1,0.2,0.05,0.3,0.1,0.6,0.6,0.7,0.8,0.99,0.8,0.67,0.5]) # prediction of model1
    c = np.array([0.3,0.6,0.2,0.1,0.1,0.9,0.23,0.7,0.9,0.4,0.77,0.3,0.89]) # prediction of model2

    AUPRC_comparison(a,b,c)
    {'auprc_ref': 0.9761904761904762,
     'auprc_target': 0.6968253968253969,
     'pvalue': 2.883157770374743e-05}
    """
    assert len(y_ref) == len(yhat_ref) == len(yhat_target)

    auprcs_ref = []
    auprcs_target = []
    
    for i in range(bootstrap):
        # sampling
        y_ref_sample, _, yhat_ref_sample, _, yhat_target_sample,_ = sklearn.model_selection.train_test_split(y_ref, yhat_ref, yhat_target, train_size=0.8, random_state=i)
        
        # compute AUPRC
        auprc_ref = sklearn.metrics.average_precision_score(y_ref_sample,yhat_ref_sample)
        auprc_target = sklearn.metrics.average_precision_score(y_ref_sample,yhat_target_sample)
        
        auprcs_ref.append(auprc_ref)
        auprcs_target.append(auprc_target)
        
    mean_ref = np.mean(auprcs_ref)
    std_ref = np.std(auprcs_ref)
    
    mean_target = np.mean(auprcs_target)
    std_target = np.std(auprcs_target)
    pvalue = pValue(mean_ref, std_ref, mean_target, std_target, bootstrap)
     
    return {'auprc_ref':auprc_ref, 'auprc_target':auprc_target, 'pvalue':pvalue}
