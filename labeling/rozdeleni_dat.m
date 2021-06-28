clc;clear all;close all;


lidi={'tom','jirina','roman','radim','laris'};


data_path='Y:\CELL\qpi_gacr\qpi_segmentation\data';
data_save_path='Y:\CELL\qpi_gacr\qpi_segmentation\data_split';


folders={'resave_pc3_2_random','resave_pnt_2_random'};
folders_save={'pc3','pnt'};


for folder_num=1:length(folders)

    folder=folders{folder_num};
    folder_save=folders_save{folder_num};


    files=subdir([data_path filesep folder filesep '*.tif']);
    files={files(:).name};
    
    for files_num=1:length(files)
        
        file=files{files_num};
    
        
        clovek_num=round((files_num)/(length(files)/(length(lidi)-1)))+1;
        

        clovek=lidi{clovek_num};

        
        mkdir([data_save_path filesep clovek filesep folder_save])
        
        file_save=file;
        
        file_save=strrep(file_save,[data_path filesep folder],[data_save_path filesep clovek filesep folder_save ]);
        
%         drawnow;
        
        copyfile(file,file_save);
        
    
    end

end