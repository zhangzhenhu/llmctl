git add .
git commit -m "ci: fix apple target by disabling cross"
git tag -d v0.1.0
git push origin --delete v0.1.0
git tag v0.1.0
git push origin v0.1.0
