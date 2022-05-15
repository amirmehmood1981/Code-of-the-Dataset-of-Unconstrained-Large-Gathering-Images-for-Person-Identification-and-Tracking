input('Please enter to proceed further....');
close all;  clear;  clc;

%Detect objects using Viola-Jones Algorithm
classification_models = {'FrontalFaceCART','FrontalFaceLBP','ProfileFace',...
    'EyePairBig','EyePairSmall','LeftEye','RightEye','LeftEyeCART','RightEyeCART',...
    'Mouth','Nose'};

cm = 2;
%FaceDetect = vision.CascadeObjectDetector('haarcascade_frontalface_default.xml','MergeThreshold',5,'MaxSize',[200,200]);
FaceDetect = vision.CascadeObjectDetector(classification_models{cm},'MergeThreshold',5,'MaxSize',[200,200]);
EyeDetect1 = vision.CascadeObjectDetector(classification_models{5},'MergeThreshold',3);
%-----------------------------[Video Sequence]-----------------------------
dir_path = 'F:Dataset Al-Nabvi Mosque\';

clip01 = 1:12;              clip02 = 17:60;             clip03 = 65:227;
clip04 = 230:366;           clip05 = 372:561;           clip06 = 568:624;
clip07 = 625:756;           clip08 = 761:932;           clip09 = 933:1036;
clip10 = 1039:1121;         clip11 = 1125:1267;         clip12 = 1271:1352;
clip13 = 1353:1383;         clip14 = 1388:1432;         clip15 = 1436:1512;
clip16 = 1513:1655;         clip17 = 1660:1756;         clip18 = 1760:1791;
clip19 = 1796:1856;         clip20 = 1859:1912;         clip21 = 1916:1938;
clip22 = 1943:2019;         clip23 = 2023:2080;         clip24 = 2084:2136;
clip25 = 2137:2208;         clip26 = 2209:2387;         clip27 = 2390:2938;         
clip28 = 2944:3222;         clip29 = 3225:3416;         clip30 = 3419:3915;
clip31 = 3919:4111;         clip32 = 4116:4485;         clip33 = 4490:4586;         
clip34 = 4589:4613;
%----------------------------[clips camera ids]----------------------------
camclip = zeros(4613,1)+nan;
camclip(clip02) = 1;
camclip(clip03) = 2;
camclip([clip04,clip17,clip24]) = 3;
camclip(clip05) = 4;
camclip([clip01,clip06,clip08,clip13,clip18,clip20]) = 5;
camclip([clip07,clip31]) = 6;
camclip(clip09) = 7;
camclip(clip10) = 8;
camclip(clip11) = 9;
camclip(clip12) = 10;
camclip(clip14) = 11;
camclip([clip15,clip16]) = 12;
camclip([clip19,clip25]) = 13;
camclip(clip21) = 14;
camclip(clip22) = 15;
camclip(clip23) = 16;
camclip(clip26) = 17;
camclip(clip27) = 18;
camclip(clip28) = 19;
camclip(clip29) = 20;
camclip(clip30) = 21;
camclip(clip32) = 22;
camclip(clip33) = 23;
camclip(clip34) = 24;

figure,
for j=1:4613    %clip07  %
    I = imread([dir_path,'RawDataFrames\IMG-',num2str(40000+j),'.png']);
    
    BB = step(FaceDetect,I);
    if isempty(BB)
        imshow(I);  
        if isnan(camclip(j))
            title(['Frame(',int2str(j),') : No face detected :: Transition']);
        else
            title(['Frame(',int2str(j),') : No face detected :: Came ID(',num2str(camclip(j)),')']);
        end
        drawnow;
    else
        IBody = insertObjectAnnotation(I,'rectangle',BB,'','Color','yellow');
        %IBody = insertObjectAnnotation(I,'rectangle',BB,'Face','Color','yellow');
        %------------------------------------------------------------------
        for k=1:size(BB,1)
            face = I(BB(k,2):BB(k,2)+BB(k,4),BB(k,1):BB(k,1)+BB(k,3),:);
            CC = step(EyeDetect1,face);
            if ~isempty(CC)
                CC(:,1:2) = CC(:,1:2) + repmat(BB(k,1:2)-1,size(CC,1),1);
                IBody = insertObjectAnnotation(IBody, 'rectangle',CC,'','Color','red');
            end
            face1 = imresize(face,[50,50],'bicubic');
            %imwrite(face,[dir_path,'Cropped Faces\IMG-',num2str(40000+j),'-',num2str(k),'.png']);
            %imwrite(face1,[dir_path,'Ehanced Faces\IMG-',num2str(40000+j),'-',num2str(k),'.png']);
        end
        imshow(IBody);
        if isnan(camclip(j))
            title(['Frame(',int2str(j),') : ',num2str(size(BB,1)),' Detected Faces (',classification_models{cm},'):: Transition']);
        else
            title(['Frame(',int2str(j),') : ',num2str(size(BB,1)),' Detected Faces (',classification_models{cm},'):: Came ID(',num2str(camclip(j)),')']);
        end
        drawnow;    %pause(0.2);
    end
end