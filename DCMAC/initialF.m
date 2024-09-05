function [G, FF, F,label] = initialF(B, num_cluster)
[G] = eig1(B, num_cluster);
stream = RandStream.getGlobalStream;
reset(stream);
G_normalized = G ./ sqrt(sum(G .^ 2, 2));
label = kmeans(G_normalized, num_cluster, 'maxiter', 1000, 'replicates', 20, 'emptyaction', 'singleton');
F = idx2pm(label);
FF = F * (F' * F + eps * eye(num_cluster)) ^ ( - 1)*F';
end


