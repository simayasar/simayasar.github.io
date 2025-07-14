---
permalink: /
title: "Tag-LLM"
author_profile: true
redirect_from: 
  - /about/
  - /about.html


---
Large Language Models (LLMs) have achieved great success in natural language processing tasks. However, this success does not easily carry over to specialized domains such as biology or chemistry. One of the main reasons is that domain-specific data — like protein sequences or chemical formulas — is underrepresented in the training data of general-purpose models.

Tag‑LLM introduces a modular and flexible approach to adapt general LLMs for specialized tasks. Instead of retraining the whole model, it guides the model using lightweight input tags that indicate the domain (e.g., protein, molecule) and the function (e.g., binding prediction, classification) of the task at hand.

These tags allow the model to understand what kind of data it is dealing with and what it is expected to do — without changing its architecture. This approach enables general LLMs to handle domain-specific tasks more effectively, and even generalize to new combinations of domains and tasks they haven’t seen before. It offers a practical path toward using general models in scientific and technical fields without building new models from scratch.


## Why Does This Problem Matter?
General-purpose large language models (LLMs) have achieved high success in natural language processing tasks. However, in specialized fields such as biology, chemistry and medicine, they do not have the same success. The main reason for this is that natural language data (text, books, articles) are often used to train these models. Since structures such as protein sequences, DNA codes or chemical formulas are quite different from natural language, the model cannot understand the context against these data and produce accurate results.

To solve this problem, some domain-specific models have been developed. For example, models for disease diagnosis, drug discovery or chemical reaction prediction. However, these models are usually trained from scratch and require a lot of labeled data and a lot of computational power. This makes them both expensive and difficult to deploy.

This is where Tag-LLM comes in. Thanks to its "domain" and "task" tags, it makes it possible to use existing generic models for different tasks without touching their architecture. In this way, they can be adapted to many different areas of expertise without the need for retraining.


## Method: What is Tag-LLM and How Does it Work?
Tag-LLM offers a modular approach to tagging, developed to guide large language models without retraining them for specific tasks. This method allows the model to adapt to different specializations and task types by adding learnable domain and function tags to the input data. The structure of the model is preserved; the labels are only placed at the input level, acting as a guide.

Below, we will explore the basic building blocks of this method: how the tags are designed, how they are integrated into the embedding layer, and how this structure contributes to the generalizability of the model.

![](/images/tag_llm_method.png)

## Soft Tag Structure
One of the core components of Tag-LLM is its soft tagging mechanism. Instead of using hard-coded text tokens, tags are represented as learnable continuous embeddings. Each tag is essentially a special vector that encodes semantic information and is directly inserted into the model’s input embedding layer.

There are two main types of tags:

**Domain Tags**: Indicate the type of specialized data the model is processing.  
For example:

`<Protein>` → protein sequences  
`<SMILES>` → chemical compound representations

**Function Tags**: Define the type of task the model should perform.  
For example:

`<BA>` → binding affinity prediction  
`<CLS>` → classification



Each tag is trained independently and used only when relevant. This means a tag can be reused across different tasks or domains. Thanks to this structure, Tag‑LLM remains both flexible and reusable, avoiding the need to retrain or redesign the core model.

## Integration into Embedding
The domain and function tags used in Tag-LLM are defined as learnable vectors (learnable embeddings), not as ordinary text tags. These tags are integrated directly into the embedding layer of the model. The aim is not to add new types of information to the model's vocabulary, but to condition task and domain knowledge into the model via the input embeddings.

Each label is a matrix of 𝑝 vectors of dimension 𝑑:

$$
\text{Tag} \in \mathbb{R}^{p \times d}
$$

The tag vector is initialized by averaging all word embeddings of the model:

$$
\hat{v} = \frac{1}{|V|} \sum_{v \in V} v
$$

The size of the tag embedding is rescaled to be consistent with other token embeddings:

$$
\text{scale} = \frac{1}{|V| \cdot \| \hat{v} \|} \sum_{v \in V} \|v\|
$$

This scaling allows the label vectors to adapt to the embedding size that the model is used to.
This learnable embedding matrix is added to the model's input sequence as if it were a special token and is processed by the transformer layers along with the other inputs. However, these tags do not appear at the output of the model, i.e. they are not generated as an output token. Instead, they only appear in the input part of the model, providing task and domain awareness in the learning process. These embeddings are optimized with gradients in the model's backpropagation process, allowing the model to learn task-specific conditionals on the input.


## How Do Tags Interact with the Model Architecture?
One of the most remarkable aspects of Tag-LLM is its ability to adapt general-purpose large language models (LLMs) to various specialized domains without changing the model’s internal structure. In traditional adaptation methods, the model’s weights are typically updated through fine-tuning or extended with new layers. In contrast, Tag-LLM operates solely through input-level tags that are added to the embedding layer.

These tags are inserted directly into the model’s input embeddings and carry task- or domain-specific information forward through the rest of the model. Importantly, the transformer layers, attention mechanisms, and other parameters of the model remain completely untouched.

As a result, the core architecture of the model remains completely frozen throughout the process. All adaptation is instead handled through the learnable tag embeddings that are added at the input level. This design allows the model to flexibly adapt to a wide range of new tasks without requiring any retraining or modification of its internal parameters.

By conditioning the model through these lightweight and modular tags—rather than altering its internals—Tag-LLM enables a powerful yet efficient form of specialization. This greatly reduces training cost while preserving the generalization strength of the original LLM.


## How Are Tags Trained? A Three-Stage Learning Strategy
Tag-LLM follows a three-stage hierarchical training protocol to adapt general-purpose large language models (LLMs) to specific domains and tasks. Through this approach, components with different functionalities, such as domain and function tags, are gradually learned from more general data to more specialized tasks.

![](/images/trainig_tag_llm.png)

### Stage 1: Learning Domain Tags
In the first phase of training, domain labels are learned, each of which belongs to a specific area of expertise. These labels sensitize the model to domains represented by external data that it has not yet encountered. For example, structures such as amino acid sequences (proteins) or chemical molecules (SMILES).

The aim is for the tag to convey general information about the domain. Thus, when the tag is included in the input, the model learns this context from the input and solves the corresponding task more efficiently.

The domain label is represented as an embedding matrix:

$$
M \in \mathbb{R}^{p \times d}
$$

The input sequence with the domain tag is defined as follows:

- Specialized domain data:  
  $$
  \{x_1, x_2, \dots, x_n\}
  $$

- Embedding representation of the data:  
  $$
  X_e \in \mathbb{R}^{n \times d}
  $$

- Input combined with the domain tag:  
  $$
  [M; X_e] \in \mathbb{R}^{(p+n) \times d}
  $$

The combined input sequence is passed to the model’s encoder layer. The model uses a **self-supervised** learning objective to predict the next token and thereby learn the domain tag:

$$
\mathcal{L}_M = \sum_{t=1}^{n} \log P(x_{t+1} \mid [M; x_1, \dots, x_t])
$$

Learning is performed based on this loss function, and the model learns to associate domain-relevant information with the tag \( M \).

This stage has two key benefits:

1. Domain-specific knowledge is directly injected into the model, making it easier to interpret specialized inputs.

2. Since the learned tag is task-agnostic, it improves the model's ability to generalize across different tasks.

### Stage 2 & 3: Learning Function Tags
One of the most powerful features of Tag‑LLM is its ability to learn function tags directly from data. These tags guide the model to perform a specific task — such as classification or regression — based solely on labeled examples. Unlike previous approaches where tasks are defined via explicit user instructions, Tag‑LLM learns the tasks implicitly from the training data itself.

During training, a domain tag is prepended to the input to indicate the type of specialized data (e.g., protein or molecule). Then, a function tag is appended to the end of the input to signal what task should be performed. This allows the model to understand both what the input represents and what it is supposed to do with it. The prediction is made based on the final hidden state associated with the function tag.

Function tags are optimized specifically for each task. For example, in a classification task, the model learns to predict the correct class label; in a regression task, it learns to output a continuous value. If the task is restricted to a single domain (e.g., protein data only), the domain tag can also be fine-tuned. However, if the task spans multiple domains (e.g., predicting interactions between proteins and molecules), domain tags are kept frozen to avoid contamination from unrelated domains — and only the function tag is trained.

These two stages work together to enable the model to perform a wide range of tasks with minimal parameter updates. Furthermore, the same function tag can be reused across domains, enabling the model to make accurate predictions even on unseen combinations of domains and tasks — a capability known as zero-shot generalization. This flexibility makes Tag‑LLM highly suitable for systems that need to handle a broad spectrum of specialized tasks.

## What If the Output Isn't Text?
Tag-LLM is not limited to text generation. In scientific tasks, the outputs that the model needs to produce are often numerical values - for example a binding score or a probability vector. Several problems arise when trying to represent such outputs in text form:

Cross-entropy loss (CE), for example, is sensitive to the similarity of numbers at the character level, not their meaning:

$$
\text{CE}(3.14, 3.24) = \text{CE}(3.14, 3.15)
$$

However, from a numerical error perspective, these are clearly not equal:

$$
\text{MSE}(3.14, 3.24) \ne \text{MSE}(3.14, 3.15)
$$

To overcome this problem, Tag-LLM uses a regression head directly instead of text output for numerical tasks. The final hidden representation of the model is multiplied by a task-specific weight matrix to produce the output:

$$
\mathbf{w}_t \in \mathbb{R}^{d \times d_t}
$$

Here, \( d \) denotes the embedding dimension of the model, and \( d_t \) is the output dimension of the task (e.g., 1 for a single numeric output, or \( k \) for a multi-class classification task).
This approach significantly improves the effectiveness of LLMs on non-natural language tasks and increases the accuracy of their outputs.

## How Does This Compare to Prior Work?
Unlike previous approaches that rely on full fine-tuning or fixed instruction prompts, Tag‑LLM introduces a modular tagging framework. This allows adaptation to both linguistic and non-linguistic domains (e.g., molecules, proteins) without changing the model’s architecture. Compared to PEFT methods like LoRA or prompt tuning, it separates domain and task knowledge via soft embeddings, enabling reusability, generalization, and efficient training.

| Feature                        | Full Fine-Tuning | Prompt Tuning | PEFT (e.g., LoRA) | **Tag-LLM**              |
|-------------------------------|------------------|----------------|-------------------|--------------------------|
| Modifies Model Architecture   | Yes           | No         | No             | No                   |
| Number of Learnable Params    | Very High     | Zero       | Medium         | Low (tags only)      |
| Task Transferability          | Poor          | Poor       | Poor           | Strong (modular)     |
| Non-linguistic Domain Support | No            | Limited    | Limited        | Yes                  |
| Training Cost                 | High          | Low        | Medium         | Low                  |
| Compatibility with LLMs       | Limited       | Moderate   | Moderate       | High                 |

## Experimental Success: How Well Does a Multilingual Translation Task Work?
To evaluate how Tag‑LLM performs in specialized domains, the first experiment was conducted on a multilingual translation task. In this setup, separate domain tags (e.g., ⟨EN⟩, ⟨FR⟩, etc.) are used for different languages, along with a single shared function tag ⟨Translate⟩ to represent the translation operation. The input format is as follows:

Input: ⟨src_lang⟩ source sentence  
Output: ⟨tgt_lang⟩ ⟨Translate⟩ target sentence

The model is trained using only tags and paired sentences—no explicit user instructions are provided. Although training data includes just 6 languages and 5 translation pairs, the model still performs competitively on unseen language pairs (e.g., ES↔PT), demonstrating the strong generalization ability of the tags.

The table below presents spBLEU scores on the FLORES-101 dataset, comparing Tag‑LLM to other models like GPT-3, BLOOM, and XGLM. Tag‑LLM ranks first on several seen pairs and yields meaningful results even on language combinations it has never seen. This shows that the ⟨Translate⟩ tag successfully encodes the translation ability, making zero-shot generalization possible.

![](/images/case_study_1.png)

## Experimental Success: Can LLMs Handle Scientific Tasks Too?
Although translation is a natural fit for LLMs, many real-world tasks—especially in science—require working with structured, non-linguistic data. Can these models generalize to such specialized domains as well? Let's find out.

### Case Study: Single-Domain, Single-Instance Tasks
To evaluate the effectiveness of Tag‑LLM in scientific tasks, experiments were first conducted on two different “single-domain, single-instance” problems. The first task involves predicting descriptor values from protein sequences; the second focuses on calculating the QED score, which indicates how drug-like a chemical compound is.

In these tasks, the inputs are not plain text — they consist of domain-specific representations such as biological sequences and chemical formulas. To enable the LLM to interpret these complex structures and make accurate numerical predictions, each task is augmented with a dedicated domain tag and a regression head. The model uses domain tags like ⟨Protein⟩ or ⟨SMILES⟩ in conjunction with task-specific tags like ⟨Descriptor⟩ or ⟨QED⟩. The structure of the input/output looks like this:

Input: The protein sequence is ⟨Protein⟩  
Output: The descriptor value is ⟨Descriptor⟩
--
Input: The SMILES of the molecule is ⟨SMILES⟩  
Output: The quantitative estimate of druglikeness is ⟨QED⟩

The model was compared against instruction-based models (e.g., GPT-4, Galactica), fine-tuning methods (e.g., LoRA, prompt tuning), and domain-specialized models (e.g., LlaSMol, Text+Chem T5). The results of these comparisons are summarized in the table below.

Tag‑LLM achieved the lowest error rates (MSE) in both the QED and descriptor tasks, outperforming not only general-purpose models but also domain-specific models with larger parameter sizes. These results highlight the effectiveness of learned input tags in adapting LLMs to scientific and structured data tasks.

![](/images/case_study_2.png)


### Case Study: Single-Domain, Multi-Instance Tasks
The second scientific benchmark for Tag‑LLM involves a regression task based on multiple inputs: drug combination prediction. In this task, the model receives two SMILES sequences representing the chemical structures of two different drugs, both preceded by the ⟨SMILES⟩ domain tag. The model is guided to produce an output using the ⟨DC⟩ function tag.

The input format is as follows:

Input: Drug 1 is ⟨SMILES⟩ ⟨input 1⟩. Drug 2 is ⟨SMILES⟩ ⟨input 2⟩
Output: The drug combination sensitivity score is ⟨DC⟩ ⟨output⟩

In this task, the model learns both the input tags and the regression head using the mean squared error (MSE) loss. Evaluation is performed using the mean absolute error (MAE) metric on the test set.

The results are summarized in the table below and are quite striking: Tag‑LLM outperforms not only powerful LLMs such as GPT-4 and Galactica but also specialized domain-specific models trained on supervised data. This demonstrates that large-scale pretrained models can leverage their general knowledge effectively to produce strong results, even in specialized domain-specific tasks.

![](/images/case_study_3.png)

### Case Study: Multi-Domain, Multi-Instance Tasks

To further evaluate Tag‑LLM’s effectiveness in scientific domains, a more challenging multi-domain regression task was introduced: predicting the binding affinity between a small molecule drug and a target protein. Traditionally, solving this task requires costly and time-consuming laboratory experiments. If LLMs can perform this prediction reliably, it could significantly accelerate drug discovery and broaden the scope of molecular screening.

In this setup, the model receives the SMILES string of the compound and the amino acid sequence of the protein, each annotated with its corresponding domain tag. The function tag ⟨BA⟩ (binding affinity) is appended to indicate the task. The format of the input is structured as follows:

Input: The protein sequence is ⟨Protein⟩ ⟨input 0⟩  
Input: The SMILES of the drug is ⟨SMILES⟩ ⟨input 1⟩  
Output: The binding affinity is ⟨BA⟩ ⟨output⟩

This explicit tagging strategy informs the model not only about the types of inputs it is processing but also the nature of the prediction it needs to perform.

To simulate a realistic distribution shift scenario, the training and testing sets are separated based on the patent year of the labels. Model performance is evaluated using Pearson correlation coefficient (r). While Tag‑LLM ranked third overall, the margin between models was very small. Importantly, Tag‑LLM achieved this result with only 86K trainable parameters, highlighting its potential to achieve even better performance with larger models. These findings underscore the strong capability of general-purpose LLMs to contribute meaningfully to scientific discovery.

![](/images/case_study_4.png)


## Why Is Tag-LLM So Effective?
It is no coincidence that Tag-LLM delivers such strong results. To understand the true impact, we need to look not only at the overall success rate, but also at the contribution of each part of the model. Systematic experiments were conducted for this: How does the model's performance change when individual components such as labels and headlines are removed? Which part is how important?

### The Contribution of Tags and the Regression Head
To better understand the components behind TAG‑LLM’s strong performance, comparisons were made to assess the impact of input tags (both domain and function tags) and the regression head. The model achieves the lowest error (MAE = 12.21) when all three components are present: a domain tag (e.g., ⟨SMILES⟩), a function tag (e.g., ⟨QED⟩), and a dedicated regression head. Removing the function tag leads to a substantial increase in error (MAE = 21.14), indicating that the model relies heavily on these tags to understand the nature of the task. Similarly, removing the regression head results in a significant drop in performance (MAE = 23.42), confirming that language modeling alone is not sufficient for precise numerical prediction. Moreover, enriching the tags with task-specific knowledge—such as augmenting the ⟨SMILES⟩ tag using QED-related information—yields even better results than using non-enriched tags. This enrichment strategy proves more effective than simply using static textual tokens like “Protein”. Altogether, these findings highlight the importance of learnable, task-aware input tags in adapting LLMs for specialized scientific applications.

These results are summarized in the table below:


| Configuration                 | MAE ↓   |
|------------------------------|---------|
| Full                         | 12.21   |
| Enriched                     | 12.10   |
| Without domain tag           | 12.39   |
| Without function tag         | 21.14   |
| Without regression head      | 23.42   |

### Effect of Tag Length
How many tokens should each tag contain? While this might seem like a minor detail, it has a direct impact on model performance. The chart below illustrates how tag length affects test error.

![](/images/tag.png)

As the tag length (p) increases, test error first decreases and then starts to rise again.

The best performance is achieved when p = 10.

This finding suggests that introducing more parameters initially helps the model, but going too far can lead to issues like overfitting.


### How Does TAG-LLM Compare to Other Techniques?
Finally, let’s take a look at how TAG-LLM performs against other PEFT (Parameter-Efficient Fine-Tuning) methods based on the results summarized in the table below:

![](/images/comparison.png)

TAG-LLM achieves the lowest error rates in both the QED and Descriptor tasks.

It also ranks first in the Drug Combination task.

For the Binding Affinity task, LoRA slightly outperforms TAG-LLM — but it's worth noting that LoRA uses approximately 12 times more learnable parameters.

These results show that TAG-LLM is highly parameter-efficient and capable of delivering strong performance even with limited resources.



## Conclusion and Future Work
Tag‑LLM shows that it’s possible to adapt frozen large language models to highly specialized domains using lightweight, modular components like input tags and a simple regression head. The results are promising even with minimal training effort.

Key Takeaways:

✅ Efficient: Achieves strong performance with few trainable parameters.

✅ Simple: No need for full fine-tuning—tags and regression head are enough.

Limitations:

⚠️ Requires better privacy, safety, and fairness considerations.

⚠️ Calls for responsible adaptation strategies, especially in sensitive domains.

Future Directions:

⏰ Applying Tag‑LLM to other scientific fields like biology, physics, or chemistry.

⏰ Exploring in-context learning to complement tag-based prompting.

## References

- [Tag-LLM: Repurposing General-Purpose LLMs for Specialized Domains (arXiv:2402.05140)](https://arxiv.org/abs/2402.05140)






















## How Important Are Input Tags and Regression Heads?
To better understand Tag-LLM’s performance, the impact of different components on the model was tested step by step. In particular, the learnable input tags (such as ⟨Protein⟩ or ⟨QED⟩) and the regression head (the component that enables numerical prediction) were individually removed to observe how the model's performance changed. The goal was to assess whether these structures are truly necessary and how much they contribute.

The table below presents the error rates and performance levels of models with different configurations. The comparisons include cases where domain information (such as the word "Protein") was only given as plain text, as well as cases where this information was paired with regression heads.

The results are clear: Tag‑LLM not only achieved the lowest error rates, but it did so with significantly fewer parameters. The improvements are especially pronounced in descriptor and QED tasks, and Tag‑LLM also outperformed all other models in the drug combination task. In the binding affinity task, although one method slightly outperformed Tag‑LLM, it should be noted that the competing model used 12 times more parameters.

This analysis demonstrates that learning domain-specific input tags and using dedicated heads for numerical tasks are critical to making large language models effective in scientific applications.

![](/images/comparison.png)

## Impact of Input Tags 
First, the effect of input tags (both domain and function tags) was individually tested on the “Drug Combination” (DC) task. When either the domain tag or the function tag was removed from the input, a significant performance drop was observed. Notably, removing the function tag caused a dramatic increase in error rate. This highlights the importance of explicitly providing task-related information to the model.

The results are summarized in the following table:

| Configuration                 | MAE ↓   |
|------------------------------|---------|
| Full                         | 12.21   |
| Enriched                     | 12.10   |
| Without domain tag           | 12.39   |
| Without function tag         | 21.14   |
| Without regression head      | 23.42   |

## Effect of Tag Length


















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
