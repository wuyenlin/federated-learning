\LoadClass{beamer}

\usetheme{Copenhagen}

\RequirePackage[brazil, english]{babel}
\RequirePackage[utf8]{inputenc}
\RequirePackage{ragged2e}
\RequirePackage{grffile}
\RequirePackage{graphicx}
\RequirePackage{amsmath,amssymb}
\RequirePackage[orientation=portrait,
    size=custom,
    width=24,
    height=29,
    scale=0.3]{beamerposter}



\RequirePackage{blindtext}

\newlength{\columnheight}
\setlength{\columnheight}{23cm}

\newenvironment{columnbox}{
    % initialize
    \begin{column}{.33\textwidth}
    \begin{minipage}[t][\columnheight]{0.95 \textwidth}
}{
    % finish
    \end{minipage}
    \end{column}
}

\providecommand{\insertsupervisor}{}
\newcommand{\supervisor}[2][\supervisorname]%
  {\renewcommand{\insertsupervisor}{#2}}
  
\beamertemplatenavigationsymbolsempty
\usebackgroundtemplate{}
\setbeamerfont{author}{size=\large}
\setbeamerfont{institute}{size=\small}
\setbeamerfont{frametitle}{size=\Huge, series=\bfseries}

\newcommand{\logoinstitute}[1]{\titlegraphic{#1}}
\newcommand{\titlebox}{
    \begin{beamercolorbox}[shadow=true, rounded=true, center]{title in head/foot}
        \usebeamerfont{frametitle}{\inserttitle}
    \end{beamercolorbox}\vfill
    \begin{minipage}[t][0.5cm][t]{\textwidth}
        \begin{minipage}[t][0.4cm][c]{.2\textwidth}
            \centering
            % \includegraphics[scale=0.2]{logo/uemasul.png}
            \inserttitlegraphic
        \end{minipage}
        \begin{minipage}[t][0.4cm][c]{.6\textwidth}
            \centering
            \usebeamerfont{author}{\insertauthor} \\
            {\insertsupervisor}\\
            \usebeamerfont{institute}{\insertinstitute}\\[2ex]
            \rule{\textwidth}{0.1px}
        \end{minipage}
        \begin{minipage}[t][0.4cm][c]{0.2\textwidth}
            \centering
            \includegraphics[height=1.2cm, keepaspectratio]{logo/logo.png}
        \end{minipage}
    \end{minipage}
}

\makeatletter
\setbeamertemplate{footline}{% override the footline from Copenhagen
  \leavevmode%
  \hbox{%
  \begin{beamercolorbox}[wd=.5\paperwidth,ht=2.25ex,dp=1ex,center]{title in head/foot}%
    \usebeamerfont{title in head/foot}\ifnum\thepage=1\insertshorttitle\else\insertframetitle\fi
  \end{beamercolorbox}%
  \begin{beamercolorbox}[wd=.5\paperwidth,ht=2.25ex,dp=1ex,center]{author in head/foot}%
    \usebeamerfont{author in head/foot}\insertshortauthor\expandafter\beamer@ifempty\expandafter{\beamer@shortinstitute}{}{~~(\insertshortinstitute)}
  \end{beamercolorbox}}%
  \vskip0pt%
}
\makeatother