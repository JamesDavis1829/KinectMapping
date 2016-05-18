x = 0;
y  = 0;
z = 0;

while(1)
    
    %import data
    file = csvread('data.txt',0,0);
    x = file(:,1);
    y = file(:,2);
    z = file(:,3);

    %graph the data
    scatter3(x,y,z);
    drawnow;
    
end