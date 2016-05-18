x = [0 0];
y  = [0 0];
z = [0 0];

while(1)
    
    %import data
    file = csvread('data.txt',0,0);
    xOld = x(2);
    yOld = y(2);
    zOld = z(2);
    x = [xOld file(end,1)];
    y = [yOld file(end,2)];
    z = [zOld file(end,3)];
    
    velocityX = norm(x(2) - x(1));
    velocityY = norm(y(2) - y(1));
    velocityZ = norm(z(2) - z(1));
    
    
     final = norm([velocityX velocityY velocityZ]);
     
     if(final > 0.001)
         disp(final);
     else
         disp(0);
     end
         
    
    
    
    pause(1);
    
end