load('log_1.mat')
len = length(recall_10)
x = 1:len
plot(x,recall_10)
hold
% load('log_2.mat')
% len = length(recall_10)
% x = 1:len
% plot(x,recall_10)

load('log_4.mat')
len = length(recall_10)
x = 1:len
plot(x,recall_10)

load('log_5.mat')
len = length(recall_10)
x = 1:len
plot(x,recall_10)

load('log_verify.mat')
len = length(recall_10)
x = 1:len
plot(x,recall_10)

load('log_ongoing.mat')
len = length(recall_10)
x = 1:len
plot(x,recall_10)

load('log_distance.mat')
len = length(recall_10)
x = 1:len
plot(x,recall_10)

legend("ICLR_2018", "SOTA + AUG", "SOTA + AUG + FSN", "SOTA + AUG + verified FSN\_partial", "SOTA + AUG + verified FSN", "SOTA + AUG + verified + distance\_candidate")

