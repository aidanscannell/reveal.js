* Repo of my reveal.js talks

** Create New Talk
To create a new talk add a new folder and org file at talks/<TALK-DIRNAME>/<TALK-FILENAME>.org.

Use '''C-c C-e R R''' to export .org file as html file using reveal.js

When generating figures using python remember to set the correct venv. (SPC SP pyvenv-workon)

** Hosting Talk on Hugo-Academic Site

1. Symlink talk directory (html file and image folder) to static/<TALK-DIRNAME>
2. Create a new talk in hugo,
'''hugo new --kind talk talk/<TALK-NAME> '''
3. Add link to reveal.js html file in talk/<TALK-DIENAME>/index.md
'''url\_slides: /talks/farscope\_seminar\_03\_20/uncertainty-ml-robotics.html'''

