files = dir('C:\Users\Guo Rui\Desktop\WiFi_CSI\csi1.dat');
len=length(files);
for k=1:len
  n=strcat('E:\CSI\data1\100\',files(k).name);
  csi_trace = read_bf_file(n);
 for i=1:200%这里是取的数据包的个数
        csi_entry = csi_trace{i};
        csi = get_scaled_csi(csi_entry); %提取csi矩阵    
        csi =csi(1,:,:);
        csi1=abs(squeeze(csi).');          %提取幅值(降维+转置)

        %只取一根天线的数据
        first_ant_csi(:,i)=csi1(:,1);           %直接取第一列数据(不需要for循环取)
        second_ant_csi(:,i)=csi1(:,2);
       % third_ant_csi(:,i)=csi1(:,3);
 end
   l=files(k).name;
   m1=strcat('E:\CSI\data1\txt\12\',l,'a.txt');
   m2=strcat('E:\CSI\data1\txt\12\',l,'b.txt');
   m3=strcat('E:\CSI\data1\txt\12\',l,'c.txt');
   dlmwrite(m1,first_ant_csi,'delimiter',' ')
   dlmwrite(m2,second_ant_csi,'delimiter',' ')
   dlmwrite(m3,third_ant_csi,'delimiter',' ')
end
 


%画第一根天线的载波
%plot(first_ant_csi.')
%plot(second_ant_csi.')
%plot(third_ant_csi.')