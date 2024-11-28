function show_clusters(AD,U,display_me)
[nclust,N] = size(U);
[sz1,sz2] = size(AD);

CL = zeros(0,sz2);


for i = 1:length(display_me),
	j = 1;
	X = zeros(0,sz2);
	for k = 1:N,
		if U(display_me(i),k) == 1,
			X(j,:) = AD(k,:);
			j = j + 1;
		end
	end
	CL = cat(1,CL,X);
	if i ~= length(display_me),
		 CL = cat(1,CL,-ones(1,sz2));
	end
end

CL = CL - min(min(CL));
CL = 64*CL/max(max(CL));
%CL = (CL+1)*32;

colormap hot

image(CL)