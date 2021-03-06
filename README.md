<h1>ULF Data Loader</h1>

<nav>
    <ol>
       <li><a href="#1">About this Project</a></li> 
       <li><a href="#2">Prerequisites</a></li>
       <li><a href="#3">How to use this project</a></li>
       <li><a href="#5">Test with ULF datasets</a></li>
        <ol>
            <li><a href="#5-1">Test with ulf-1.0</a></li>
            <li><a href="#5-2">Test with ulf-1.0-stog</a></li>
        </ol>
        <li><a href="#6">Associated Links</a></li>
    </ol>
</nav>

<div id="1">
    <h2>About this Project</h2>
    <p>
        This Project is a data loader for the ULF dataset.
        <a href="https://www.cs.rochester.edu/u/gkim21/ulf/ulf/">
            ULF dataset
        </a> is a NLP project at the University of Rochester.
        <b>This project is not available for Windows as Allen NLP do not have Windows support yet.</b>
    </p>
    <ul>
        <li>loader.py -> holds a function that returns data loaders from ULF data set</li>
        <li>ULFreader.py -> holds a dataset reader class for ULF dataset</li>
    </ul>
    <p>
    It gives you vocabularies and data loaders for raw sentence, ULF-proprocessed and AMR processed data.
    Using this, anyone can train their model.
    ULF dataset has four columns.
    <ul>
        <li>ID : dataset ID</li>
        <li>sentence : raw sentences</li>
        <li>ULF : ULF preprocessed data</li>
        <li>ULF-AMR : AMR processed data</li>
    </ul>
    This project produces separate vocabularies and namespaces for each label except ID.
    </p>
</div>
</div>

<div id="2">
    <h2>Prerequisites</h2>
    <p>
        This project is built with Allen NLP.
        <a href="https://docs.allennlp.org/main/#getting-started-using-the-library">source of the installation guide</a>.
        <ul>
            <li>Conda</li>
            <li>Python3</li>
            <li>AllenNLP</li>
        </ul>
        Please run the following command after runnning the conda virtual environment.
    </p>
        
        conda create allennlp-env
        conda activate allennlp-env
        pip3 install allennlp
</div>

<div id="3">
    <h2>How to use this project</h2>
    Please refer to the following code.

    data_loader2 = loader(True) #ulf-1.0 dataset
    data_loader2 = loader(False) #ulf1-1.0-stog dataset
</dvi>

<div id="5">
    <h2>ULF datasets</h2>
</div>

<div id="5-1">
    <h3>Test with ulf-1.0</h3>
    The dataset of the first official release of annotated ULFs.
    The dataset is a list of [id, sentence, ULF, ULF-AMR] entries.
</div>

<div id="5-2">
    <h3>Test with ulf-1.0-stog</h3>
    A version of ulf-1.0.json with additional preprocessing steps to
    make it compatible with the sequence-to-graph AMR parser.
</div>

<div id="6">
    <h2>Associated Links</h2>
    <ul>
        <li>ULF dataset https://www.cs.rochester.edu/u/gkim21/ulf/ulf/</li>
        <li>Allen NLP Docs https://docs.allennlp.org/main/#getting-started-using-the-library</li>
    </ul>
</div>


