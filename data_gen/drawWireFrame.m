function [] = drawWireFrame(pts, color)

pts = pts';
i = 1 ;
while (i <= size(pts,1))
    for j=1:1:4
        if ( j ~= 4)
            line_x = [pts(i+j-1,1), pts(i+j,1)];
            line_y = [pts(i+j-1,2), pts(i+j,2)];
            line_z = [pts(i+j-1,3), pts(i+j,3)];
        else
            line_x = [pts(i+j-1,1), pts(i+j-4,1)];
            line_y = [pts(i+j-1,2), pts(i+j-4,2)];
            line_z = [pts(i+j-1,3), pts(i+j-4,3)];
        end  
        plot3(line_x, line_y, line_z, color); hold on; 
    end
    i = i+4;
end

for i=1:1:4
    line_x = [pts(i,1), pts(i+4,1)];
    line_y = [pts(i,2), pts(i+4,2)];
    line_z = [pts(i,3), pts(i+4,3)];
    plot3(line_x, line_y, line_z, color); hold on; 
end

grid on; grid minor; 
xlim([-100 350])
ylim([-100 350])
zlim([-100 350])
end

