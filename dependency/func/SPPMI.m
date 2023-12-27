function [ SPPMI ] = createSPPMIMtx(G , k)

  nodeDegrees = sum(G);   %???????
  nodeDegrees2=sum(G,2);  %???????
  W = sum(nodeDegrees);   %????
  SPPMI = G;
% use a loop to calculate Wij*W/(di*dj)
  [col,row,weights] = find(G);
  for i = 1:length(col)
          score = log(weights(i) * W / nodeDegrees2(col(i)) / nodeDegrees(row(i))) - log(k);
          if(score > 0)
            SPPMI(col(i),row(i)) = score;
          else
            SPPMI(col(i),row(i)) = 0;
         end
  end

%   spfun(@shiftOpt,SPPMI);
%     function score = shiftOpt(x)
%         score = log(x) - log(k);
%         if(score<0)
%             score = 0;
%         end
%     end
%     disp('SPPMI Matrix is??Done');
end

