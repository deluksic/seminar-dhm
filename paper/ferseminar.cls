\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ferseminar}[2011/04/14 Predlozak za seminar na FER-u]
\LoadClass[a4paper,pdflatex,twoside,twocolumn]{article}

%----------------------------------------------------------------------------------------
%	PACKAGES AND OTHER DOCUMENT CONFIGURATIONS
%----------------------------------------------------------------------------------------

\usepackage{blindtext} % Package to generate dummy text throughout this template 
\usepackage{mathtools}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{units}
\lstset{language=C,keywordstyle={\bfseries \color{blue}},basicstyle={\small\ttfamily}}
\definecolor{commentcolor}{RGB}{0,127,0}
\lstdefinestyle{Python}{
	language        = Python,
	basicstyle      = \small\ttfamily,
	keywordstyle    = \color{blue},
	stringstyle     = \color{red},
	commentstyle    = \color{commentcolor}\ttfamily
}
\usepackage{graphicx}
\graphicspath{ {images/} }

\usepackage[sc]{mathpazo} % Use the Palatino font
\usepackage[T1]{fontenc} % Use 8-bit encoding that has 256 glyphs
\linespread{1.2} % Line spacing - Palatino needs more space between lines
\usepackage{microtype} % Slightly tweak font spacing for aesthetics

\usepackage[utf8]{inputenc}
\usepackage[croatian]{babel}

\usepackage[top=25mm, bottom=25mm, left=20mm, right=20mm, columnsep=6mm]{geometry}
\usepackage[hang, small,labelfont=bf,up,textfont=it,up]{caption}
\usepackage{booktabs} % Horizontal rules in tables

\usepackage{lettrine} % The lettrine is the first enlarged letter at the beginning of the text

\usepackage{enumitem} % Customized lists
\setlist[itemize]{noitemsep} % Make itemize lists more compact
\setlist[enumerate]{itemsep=0mm}

\usepackage{abstract} % Allows abstract customization
\renewcommand{\abstractnamefont}{\normalfont\bfseries} % Set the "Abstract" text to bold
\renewcommand{\abstracttextfont}{\normalfont\small\itshape} % Set the abstract itself to small italic text

\usepackage{titlesec} % Allows customization of titles
\renewcommand\thesection{\Roman{section}} % Roman numerals for the sections
\renewcommand\thesubsection{\roman{subsection}} % roman numerals for subsections
\titleformat{\section}[block]{\large\scshape\centering}{\thesection.}{1em}{} % Change the look of the section titles
\titleformat{\subsection}[block]{\large}{\thesubsection.}{1em}{} % Change the look of the section titles

\usepackage{fancyhdr} % Headers and footers
\pagestyle{fancy} % All pages have headers and footers
\fancyhead{} % Blank out the default header
\fancyfoot{} % Blank out the default footer
\fancyhead[C]{\Naslov\ $\bullet$ \MjestoDatum} % Custom header text
\fancyfoot[RO,LE]{\thepage} % Custom footer text

\usepackage{titling} % Customizing the title section

\usepackage{hyperref} % For hyperlinks in the PDF

%----------------------------------------------------------------------------------------
%	TITLE SECTION
%----------------------------------------------------------------------------------------

% ovo su definicije varijabli koje treba popuniti
\newcommand{\student}[1]{\def \Student {#1}}
\newcommand{\voditelj}[1]{\def \Voditelj {#1}}
\newcommand{\mjestodatum}[3]{\def \MjestoDatum {#1, #2, #3.}}
\newcommand{\naslov}[1]{\def \Naslov {#1}}
\newcommand{\sazetak}[1]{\def \Sazetak {#1}}
\newcommand{\email}[1]{\def \Email {#1}}

\renewcommand{\maketitlehookd}{%
	\begin{abstract}
		\textbf{\noindent \Sazetak}
	\end{abstract}
}

%----------------------------------------------------------------------------------------

% ovo je naredba za stvaranje naslova
\newcommand{\umetninaslov}{
	\setlength{\droptitle}{-4\baselineskip} % Move the title up
	
	\pretitle{\begin{center}\Huge\bfseries} % Article title formatting
		\posttitle{\end{center}} % Article title closing formatting
	\title{\Naslov} % Article title
	\author{%
		\textsc{\Student} \\
		\textsc{voditelj: \Voditelj} \\[1ex]
		\normalsize FAKULTET ELEKTROTEHNIKE I RAČUNARSTVA \\
		\normalsize \Email \\
	}
	\date{\today} % Leave empty to omit a date
	\maketitle
}