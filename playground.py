from tqdm import tqdm
import time

list = [10, 20, 30, 40, 50, 60, 70]

for i in tqdm(list, total=len(list)):
    time.sleep(5)
    print(i)
    

