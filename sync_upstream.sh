git checkout main
git remote add pytorch-upstream https://github.com/pytorch/pytorch
git fetch pytorch-upstream main
git push --force origin refs/remotes/pytorch-upstream/main:refs/heads/main

