function [] = drawimage(pts, color, img_size)



pts = pts';
i = 1 ;
while (i <= size(pts,1))
    for j=1:1:4
        if ( j ~= 4)
            line_x = [pts(i+j-1,1), pts(i+j,1)];
            line_y = [pts(i+j-1,2), pts(i+j,2)];
        else
            line_x = [pts(i+j-1,1), pts(i+j-4,1)];
            line_y = [pts(i+j-1,2), pts(i+j-4,2)];
        end  
        plot(line_x, line_y, color); hold on; 
    end
    i = i+4;
end

for i=1:1:4
    line_x = [pts(i,1), pts(i+4,1)];
    line_y = [pts(i,2), pts(i+4,2)];
    plot(line_x, line_y, color); hold on; 
end

xlim([0 img_size(1)])
ylim([0 img_size(2)])
end

