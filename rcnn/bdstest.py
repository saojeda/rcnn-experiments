from statsmodels.tsa.stattools import bds
from rcnn.dataloader import CNNDataLoader
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    regression_datasets = ["BenzeneConcentration",
                           "HouseholdPowerConsumption1",
                           "NewsHeadlineSentiment",
                           "BIDMC32HR",
                           "LiveFuelMoistureContent",
                           "IEEEPPG",
                           "PPGDalia"]

    for name in regression_datasets:
        print(name)
        dataloader = CNNDataLoader(path_to_file="/home/sun/Documents/Monash_UEA_UCR_Regression_Archive/"
                                                f"{name}/{name}_TEST.ts", img_size=4)
        residuals = dataloader.get_residuals()
        for i in range(residuals.shape[0]):
            _, pval = bds(residuals[i])
            print(pval)
