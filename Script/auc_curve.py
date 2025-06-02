import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# Define the file paths
# Result path dir
base_path = './result_path_dir/'
methods = ['andt', 'astt', 'ffp', 'mlep', 'mnad', 'std']

output_path = os.path.join('./visualization', 'roc_curves_output.png')


fpr_data = {}
tpr_data = {}
auc_data = {}

# Load FPR, TPR, and AUC values from npy files for each method
for method in methods:
    fpr_data[method] = np.load(base_path + f'{method}_fpr.npy')
    tpr_data[method] = np.load(base_path + f'{method}_tpr.npy')
    # Calculate AUC using FPR and TPR
    calculated_auc = auc(fpr_data[method], tpr_data[method])
    auc_data[method] = calculated_auc

# Plot ROC curves
plt.figure(figsize=(10, 9))

for method in methods:
    labelMethod = method.upper()   
    plt.plot(fpr_data[method], tpr_data[method], label=f'{labelMethod} (AUC = {auc_data[method]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curves for UIT-ADrone', fontsize=18)
plt.legend(loc='lower right', fontsize=18)

# Display the plot
plt.show()

# Save the plot as a PNG file
plt.savefig(output_path, dpi=1024, bbox_inches='tight')
plt.close()
