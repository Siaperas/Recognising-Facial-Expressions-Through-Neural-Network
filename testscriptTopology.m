A = importdata('cleandata_students.mat'); 
B = importdata('noisydata_students.mat');

lr = 0.065; 
mom = 0.475; 

topology = [ 10 10; 20 20; 25 25; 30 30; 35 35; 40 40; 50 50; 65 65; 80 80; 95 95]; 

for i=1:10 
    disp(i)
    [BestPerf,BestErr,ErrNet,PerfNet,ErrInd,PerfInd] = TrainNetwork(A.x,A.y,  topology(i,:),lr,0,0,mom, 0, 0);
    best(i) = BestPerf;
    beste(i)= BestErr;
end
figure(1)
plot(topology, best); 

topology = [ 10; 20; 25; 30; 35; 40; 50; 65; 80; 95]; 

for i=1:10 
    disp(i)
    [BestPerf,BestErr,ErrNet,PerfNet,ErrInd,PerfInd] = TrainNetwork(A.x,A.y,  topology(i),lr,0,0,mom,0,0);
    best(i) = BestPerf;
    beste(i)= BestErr;
end
figure(2)
plot(topology, best); 

