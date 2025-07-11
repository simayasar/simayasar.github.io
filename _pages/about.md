---
permalink: /
title: "Tag-LLM: Repurposing General-Purpose LLMs for Specialized Domains"
author_profile: true
redirect_from: 
  - /about/
  - /about.html


---


## Why Do General LLMs Fall Short?

While general-purpose large language models (LLMs) have shown impressive performance on natural language tasks, this success often does not carry over to technical domains such as biology and chemistry. The core issue lies in their training data: these models are typically trained on natural language corpora like Wikipedia, books, or news articles, which lack the specialized syntax, symbols, and structural representations found in scientific data — such as protein sequences or SMILES formulas.

Moreover, tasks in these domains require more than just linguistic understanding; they demand domain-specific reasoning and knowledge. Since general LLMs are not exposed to such structured inputs or expert-level semantics, they struggle to accurately interpret and infer in these contexts.

Tag-LLM addresses these limitations by proposing a modular and flexible approach that enhances general-purpose LLMs for specialized tasks — without retraining them from scratch or altering their core architecture.



## Why Does This Problem Matter?

While general-purpose large language models (LLMs) perform well in natural language processing tasks, their performance degrades significantly when applied to specialized domains (biology, chemistry, medicine, etc.). The main reason for this is that these models are largely trained with natural language datasets. However, specialized formats such as protein sequences, DNA codes or SMILES representations of chemical compounds are structurally and symbolically quite different from natural language. Since such data is not adequately represented in the training process of general LLMs, the model fails to understand the context and make correct inferences.

To fill this gap, models have been developed for specific domains, such as specialized LLMs trained for disease diagnosis systems, chemical synthesis prediction or drug discovery. However, these models are usually trained from scratch and require a huge amount of labeled data and computational resources. This makes for an approach that is both costly and unsustainable in terms of scalability. 

This naturally leads to the question: Is there a way to adapt the powerful general-purpose LLMs we already have to specialized tasks without disrupting their internal structure or retraining them? Tag‑LLM offers a clear answer to this question. By placing “domain” and “function” tags at the input level, general LLMs can be directed toward different tasks and data types. This approach eliminates the need for retraining and enables these models to perform effectively across a much broader range of tasks.









Getting started
======
1. Register a GitHub account if you don't have one and confirm your e-mail (required!)
1. Fork [this template](https://github.com/academicpages/academicpages.github.io) by clicking the "Use this template" button in the top right. 
1. Go to the repository's settings (rightmost item in the tabs that start with "Code", should be below "Unwatch"). Rename the repository "[your GitHub username].github.io", which will also be your website's URL.
1. Set site-wide configuration and create content & metadata (see below -- also see [this set of diffs](http://archive.is/3TPas) showing what files were changed to set up [an example site](https://getorg-testacct.github.io) for a user with the username "getorg-testacct")
1. Upload any files (like PDFs, .zip files, etc.) to the files/ directory. They will appear at https://[your GitHub username].github.io/files/example.pdf.  
1. Check status by going to the repository settings, in the "GitHub pages" section

Site-wide configuration
------
The main configuration file for the site is in the base directory in [_config.yml](https://github.com/academicpages/academicpages.github.io/blob/master/_config.yml), which defines the content in the sidebars and other site-wide features. You will need to replace the default variables with ones about yourself and your site's github repository. The configuration file for the top menu is in [_data/navigation.yml](https://github.com/academicpages/academicpages.github.io/blob/master/_data/navigation.yml). For example, if you don't have a portfolio or blog posts, you can remove those items from that navigation.yml file to remove them from the header. 

Create content & metadata
------
For site content, there is one Markdown file for each type of content, which are stored in directories like _publications, _talks, _posts, _teaching, or _pages. For example, each talk is a Markdown file in the [_talks directory](https://github.com/academicpages/academicpages.github.io/tree/master/_talks). At the top of each Markdown file is structured data in YAML about the talk, which the theme will parse to do lots of cool stuff. The same structured data about a talk is used to generate the list of talks on the [Talks page](https://academicpages.github.io/talks), each [individual page](https://academicpages.github.io/talks/2012-03-01-talk-1) for specific talks, the talks section for the [CV page](https://academicpages.github.io/cv), and the [map of places you've given a talk](https://academicpages.github.io/talkmap.html) (if you run this [python file](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.py) or [Jupyter notebook](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.ipynb), which creates the HTML for the map based on the contents of the _talks directory).

**Markdown generator**

The repository includes [a set of Jupyter notebooks](https://github.com/academicpages/academicpages.github.io/tree/master/markdown_generator
) that converts a CSV containing structured data about talks or presentations into individual Markdown files that will be properly formatted for the Academic Pages template. The sample CSVs in that directory are the ones I used to create my own personal website at stuartgeiger.com. My usual workflow is that I keep a spreadsheet of my publications and talks, then run the code in these notebooks to generate the Markdown files, then commit and push them to the GitHub repository.

How to edit your site's GitHub repository
------
Many people use a git client to create files on their local computer and then push them to GitHub's servers. If you are not familiar with git, you can directly edit these configuration and Markdown files directly in the github.com interface. Navigate to a file (like [this one](https://github.com/academicpages/academicpages.github.io/blob/master/_talks/2012-03-01-talk-1.md) and click the pencil icon in the top right of the content preview (to the right of the "Raw | Blame | History" buttons). You can delete a file by clicking the trashcan icon to the right of the pencil icon. You can also create new files or upload files by navigating to a directory and clicking the "Create new file" or "Upload files" buttons. 

Example: editing a Markdown file for a talk
![Editing a Markdown file for a talk](/images/editing-talk.png)

For more info
------
More info about configuring Academic Pages can be found in [the guide](https://academicpages.github.io/markdown/), the [growing wiki](https://github.com/academicpages/academicpages.github.io/wiki), and you can always [ask a question on GitHub](https://github.com/academicpages/academicpages.github.io/discussions). The [guides for the Minimal Mistakes theme](https://mmistakes.github.io/minimal-mistakes/docs/configuration/) (which this theme was forked from) might also be helpful.
