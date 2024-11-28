function show_clusters(AD,U)
[nclust,N] = size(U);
[N,n]=size(AD); 
figure
display_me = 1;
while display_me
	display_me = input('Which cluster would you like to see? [0 exits]:');

	if display_me > 0
		if display_me > nclust
			fprintf('there are only %d clusters\n', nclust)
		else
			CL = AD(1,:);
			j = 1;
			for k = 1:N,
				if U(display_me,k) == 1,
					CL(j,:) = AD(k,:);
					j = j + 1;
				end
			end
			plot(1:n,CL)
			[sz1,sz2] = size(CL);
			T = sprintf('Cluster %d with %d vectors', display_me, sz1);
			title(T)
		end
	end
end
