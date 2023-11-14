import numpy as np
import os
import re
def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                L.append(os.path.join(root, file))
    return L

for k in range(10):
    m=k+1
    mi=str(m)
    ni = str(11)
    filename = 'C:/Users/Guo Rui/Desktop/WiFi_CSI/txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
    a=file_name(filename)
    for l in range(2):#此处的数字为文件夹中样本的数量
      filenames=a[l]
      pos = []
      Efield = []
      with open(filenames, 'r') as file_to_read:
        while True:
          lines = file_to_read.readline() # 整行读取数据
         # print(lines)
          if not lines:
            break
            pass
          p_tmp= [float(i) for i in lines.split()]
          E_tmp = [float(i) for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
          pos.append(p_tmp)  # 添加新读取的数据
          Efield.append(E_tmp)
          pass
      pos = np.array(pos) # 将数据从list类型转换为array类型。
      Efield = np.array(Efield)
      str1=r'txt'
      str2=r'npy'
      tt = re.sub(str1, str2, filenames)
      np.save(tt,pos)
      print(tt)
      pass
