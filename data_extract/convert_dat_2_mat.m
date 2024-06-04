clear;
clc;
close all;

date_select=""

root_dir = ""

out_dir="";
if ~isfolder(out_dir)
    mkdir(out_dir);
end

if date_select == "20181109"
    room=1
     users = ["user1","user2","user3"];
    ges_ids=["Push&Pull","Sweep","Clap","Slide","Draw-Zigzag(Vertical)","Draw-N(Vertical)"];
end

% traverseFolder(root_dir);

mm=1;
for m = 1:numel(users)
    user = users(m);
    disp(user);
    subfolder = fullfile(root_dir,user);
    
    
    files = dir(subfolder);
    for j = 1:numel(files)
        if files(j).name(1) == '.'
            % 忽略当前目录和上级目录的文件夹
            continue;
        end

        if files(j).isdir
            % 如果是文件夹，则递归遍历子文件夹
            continue;
        else
            % 如果是文件，则执行相应操作    'room_3_user_3_ges_Push&Pull_loc_1_ori_1_rx_1_csi.mat'
           file_path = fullfile(subfolder, files(j).name);
           
           info = strsplit(files(j).name, '-');
           ges=str2num(info{2});
           loc=info{3};
           ori=info{4};
           rep=info{5};
           rx=info{6}(2);
           %
           if ges >numel(ges_ids)
                disp(file_path+'，不在手势范围内');
               continue;
           end 
           out_filename = ['room_', room, '_user_', regexp(user, '\d+', 'match'), '_ges_',ges_ids(ges) , '_loc_', loc, '_ori_', ori, '_rx_', rx, '_rep_', rep,'_date_',date_select,'_csi.mat'];
           out_filename_str = join(out_filename, '');
           out_path=fullfile(out_dir,out_filename_str);
           if exist(out_path, 'file')
               
               disp(file_path+'，文件已存在，跳过创建步骤, '+out_path+" ,累计"+mm);
               mm=mm+1;
               continue;
           end
           
           try
               disp(file_path+",输出路径，"+out_path);
               [csi_data, ~] = csi_get_all(file_path);
               save(out_path,'csi_data');
           catch exception
               warning('An error occurred while processing the data:\n%s', exception.message);
           end
        end
    end
       
end



