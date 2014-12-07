stat-nlp
========

This is a repository for the homework assignments of Statistical Natural Language Processing.

To run
------------------
First, make sure that the directory `bionlp2011genia-train-clean` is in the same directory as `assignment2.py`. For a short demonstration, run the following command
```
python assignment2.py
```

Workflow
------------------

1. Anything in the master branch is deployable
2. To work on something new, create a descriptively named branch off of master (ie: new-oauth2-scopes)
3. Commit to that branch locally and regularly push your work to the same named branch on the server
4. When you need feedback or help, or you think the branch is ready for merging, open a pull request
5. After someone else has reviewed and signed off on the feature, you can merge it into master
6. Once it is merged and pushed to ‘master’, you can and should deploy immediately

<small>Source: [Scott Chacon - GitHub Flow](http://scottchacon.com/2011/08/31/github-flow.html)</small>

You can follow this workflow like this:

```
git branch feature-or-fix-description
git checkout feature-or-fix-description
# make some changes
git add .
git commit -m "Add this" # start with a verb in present tense
git push origin feature-or-fix-description
# open pull request on github
# merge & close PR on github
git checkout master
git branch -D feature-or-fix-description
git pull origin master
```

Python Style Conventions
------------------

We adhere to the Python styling conventions as outlined by the <small>[Google Python Style Guide](https://google-styleguide.googlecode.com/svn/trunk/pyguide.html)</small>
