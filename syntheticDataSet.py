import random
import pandas as pd
from datetime import datetime
def generate(VM,timestamp):
    cpu_usage=random.randint(0, 100)
    memory_usage=random.randint(0, 100)
    latency=round(random.uniform(0,200), 3)
    return {"VM Name": f"VM{VM}","cpu_usage": cpu_usage, "memory_usage":memory_usage, "latency":latency, "timestamp":timestamp}
liste=[]
now=datetime.now()
for k in range(40):
    for j in range(1,31):
        for i in range(0,23):
            liste.append(generate(k,datetime(now.year, now.month-1, j, i)))
df=pd.DataFrame(liste).set_index("VM Name")
df.to_csv(r"metric_monthly2.csv")