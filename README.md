# NetGP


Drug response prediction (DRP) is important for precision medicine to predict how a patient would react to a drug before administration. 
Existing studies take the cell line transcriptome data, and the chemical structure of drugs as input and predict drug response as IC50 or AUC values. 
Intuitively, use of drug target interaction (DTI) information can be useful for DRP.
However, use of DTI is difficult because existing drug response database such as CCLE and GDSC do not have information about transcriptome after drug treatment.
Although transcriptome after drug treatment is not available, if we can compute the perturbation effects by the pharmacologic modulation of target gene, we can utilize the DTI information in CCLE and GDSC.
In this study, we proposed a framework that can improve existing deep learning-based DRP models by effectively utilizing drug target information.
Our framework includes NetGP, a module to compute gene perturbation scores by the network propagation technique on a PPI network.
NetGP produces genes in a ranked list in terms of gene perturbation scores and the ranked genes are input to a MLP to generate a fixed dimension vector for the integration with existing DRP models.
This integration is done in a model-agnostic way so that any existing DRP tool can be incorporated.
As a result, our framework boosts the performance of existing DRP models, in 44 of 48 comparisons. The performance gains are larger especially for test scenarios with samples with unseen drugs by large margins up to 34% in Pearson correlation coefficient.
