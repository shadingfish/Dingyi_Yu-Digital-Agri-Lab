import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)

# 加载.npy文件
data = np.load(
    r'C:\Users\Administrator\Desktop\training data\small\test_0_0.npy'
)
# 查看前50行数据
print(data[:50, :])
print("Finishing.")

data_transposed = data.T
# Convert the transposed data to a Pandas DataFrame
df = pd.DataFrame(data_transposed)
# Save the DataFrame to a .csv file
csv_file_path = r'C:\Users\Administrator\Desktop\training data\small\transposed_data.csv'
df.to_csv(csv_file_path, index=False)



