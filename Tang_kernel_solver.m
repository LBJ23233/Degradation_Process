function kernels = Tang_kernel_solver(hr_path,lr_path)
% Kernel estimation

kernel_radius = 8;
kernel_size = 2*kernel_radius+1;
hr=im2double(imread(hr_path));
lr=im2double(imread(lr_path));

% crop patch
num=4;
size_hr=size(hr);
size_lr=size(lr);
scale=size_hr(1)/size_lr(1);
crop_size=size_hr/sqrt(num);
hr_patch=zeros(num, crop_size(1), crop_size(2), 3);
lr_patch=zeros(num, crop_size(1)/scale, crop_size(2)/scale, 3);
kernels =zeros(num, kernel_size, kernel_size);
idx=0;
for i=0:sqrt(num)-1
    for j=0:sqrt(num)-1
        idx = idx+1;
        lr_patch(idx,:,:,:)=lr(i*crop_size(1)/scale+1:i*crop_size(1)/scale+crop_size(1)/scale, j*crop_size(2)/scale+1:j*crop_size(2)/scale+crop_size(2)/scale, :);
        hr_patch(idx,:,:,:)=hr(i*crop_size(1)+1:i*crop_size(1)+crop_size(1), j*crop_size(2)+1:j*crop_size(2)+crop_size(2), :);
    end
end
size_p=size(lr_patch);
for idx=1:size_p(1)
    hr = squeeze(hr_patch(idx,:,:,:));
    lr = squeeze(lr_patch(idx,:,:,:));

    %detect texture region (corners) and avoid boundary corners
    gray=rgb2gray(lr(kernel_radius:end-kernel_radius,kernel_radius:end-kernel_radius,:));
    [l_height,l_width]=size(gray);
    corners = corner(gray, 5000, 'QualityLevel',0.001, 'SensitivityFactor',0.001);
    size_corners = size(corners);

    %detect edge and combine(optional)
    edges=edge(gray,'canny');
    [grid_x,grid_y]=meshgrid([1:l_width],[1:l_height]);
    grid_x=grid_x(edges>0.5);
    grid_y=grid_y(edges>0.5);
    size_edge=size(grid_x);

    size_corners=size_corners+size_edge(1);
    corners=[corners;[grid_x,grid_y]];

    %shift back to original coordinate
    corners(:,1)=corners(:,1)+kernel_radius;
    corners(:,2)=corners(:,2)+kernel_radius;


    %construct matrix for minimize|k*y-x|
    C=zeros(3*size_corners(1)+kernel_size*kernel_size, kernel_size*kernel_size);
    d=zeros(3*size_corners(1)+kernel_size*kernel_size, 1);
    for i=1:size_corners(1)
        for c=1:3
            for y=-1:0
                for x=-1:0
                    C(3*(i-1)+c,:)= C(3*(i-1)+c,:)+reshape(hr(2*corners(i,2)-kernel_radius+y:2*corners(i,2)+kernel_radius+y,...
                                                              2*corners(i,1)-kernel_radius+x:2*corners(i,1)+kernel_radius+x,c),1,[]);
                end
            end
            C(3*(i-1)+c,:) = C(3*(i-1)+c,:)/4;
            d(3*(i-1)+c,:) = lr(corners(i,2),corners(i,1),c);
        end
    end


    % matrix for non negative constraint
    A = -eye(kernel_size*kernel_size, kernel_size*kernel_size);
    b = zeros(kernel_size*kernel_size, 1);

    %solve kernel
    k = lsqlin(C, d, A, b);

    %reshape and visualize
    max(k);
    k = reshape(k,[kernel_size,kernel_size]);
    kernels(idx,:,:) = k;
end
end
