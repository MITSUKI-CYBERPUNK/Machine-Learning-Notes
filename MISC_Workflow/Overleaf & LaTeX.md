# 1 LaTeX
+ LaTeX是一种标记语言，主要用于创建高质量的学术文档，特别是数学、物理和计算机科学领域的文档。它基于TeX排版系统，由美国数学家Donald E. Knuth开发。在LaTeX中，你可以轻松地编写复杂的数学公式，并控制文档的布局和样式。

[[Markdown基础语法#3. 内联 $ LaTeX$ 公式 [用 "$" 包围]]]
## 1.1 两种方式

在LaTeX中编辑数学公式的两种表现形式：

- 行内：使用一对`$公式部分$`即可，比如勾股定理的公式：`$a^2 +b^2=c^2$`
- 行外：使用两对`$$公式部分$$`，勾股定理`$$a^2 +b^2=c^2$$`会**单独占据一行**

## 1.2 常见字母表
### 1.2.1 常用希腊字母 Greek alphabet

希腊字母是希腊语所使用的字母，也广泛用于数学、物理、生物、化学、天文等学科。它是世界上最早拥有表示元音音位的字母的书写系统

| 序号  | 小写  | LaTeX       | 读音                     | 序号  | 大写  | LaTeX     | 读音          |
| :-: | :-: | :---------- | :--------------------- | :-: | :-: | :-------- | :---------- |
|  1  |  α  | \alpha      | /ˈælfə/                | 22  |  σ  | \sigma    | /ˈsɪɡmə/    |
|  2  |  β  | \beta       | /ˈbiːtə/, US: /ˈbeɪtə/ | 23  |  ς  | \varsigma | /ˈsɪɡmə/    |
|  3  |  γ  | \gamma      | /ˈɡæmə/                | 24  |  τ  | \tau      | /taʊ, tɔː/  |
|  4  |  δ  | \delta      | /ˈdɛltə/               | 25  |  υ  | \upsilon  | /ˈʌpsɪlɒn/  |
|  5  |  ϵ  | \epsilon    | /ˈɛpsɪlɒn/             | 26  |  ϕ  | \phi      | /faɪ/       |
|  6  |  ε  | \varepsilon | /ˈɛpsɪlɒn/             | 27  |  φ  | \varphi   | /faɪ/       |
|  7  |  ζ  | \zeta       | /ˈzeɪtə/               | 28  |  χ  | \chi      | /kaɪ/       |
|  8  |  η  | \eta        | /ˈeɪtə/                | 29  |  ψ  | \psi      | /psaɪ/      |
|  9  |  θ  | \theta      | /ˈθiːtə/               | 30  |  ω  | \omega    | /oʊˈmeɪɡə/  |
| 10  |  ϑ  | \vartheta   | /ˈθiːtə/               | 31  |  Γ  | \Gamma    | /ˈɡæmə/     |
| 11  |  ι  | \iota       | /aɪˈoʊtə/              | 32  |  Δ  | \Delta    | /ˈdɛltə/    |
| 12  |  κ  | \kappa      | /ˈkæpə/                | 33  |  Θ  | \Theta    | /ˈθiːtə/    |
| 13  |  λ  | \lambda     | /ˈlæmdə/               | 34  |  Λ  | \Lambda   | /ˈlæmdə/    |
| 14  |  μ  | \mu         | /mjuː/                 | 35  |  Ξ  | \Xi       | /zaɪ, ksaɪ/ |
| 15  |  ν  | \nu         | /njuː/                 | 36  |  Π  | \Pi       | /paɪ/       |
| 16  |  ξ  | \xi         | /zaɪ, ksaɪ/            | 37  |  Σ  | \Sigma    | /ˈsɪɡmə/    |
| 17  |  o  | o           | /ˈɒmɪkrɒn/             | 38  |  Υ  | \Upsilon  | /ˈʌpsɪlɒn/  |
| 18  |  π  | \pi         | /paɪ/                  | 39  |  Φ  | \Phi      | /faɪ/       |
| 19  |  ϖ  | \varpi      | /paɪ/                  | 40  |  Ψ  | \Psi      | /psaɪ/      |
| 20  |  ρ  | \rho        | /roʊ/                  | 41  |  Ω  | \Omega    | /oʊˈmeɪɡə/  |
| 21  |  ϱ  | \varrho     | /roʊ/                  |     |     |           |             |
|     |     |             |                        |     |     |           |             |

### 1.2.2 常用希伯来字母Hebrew alphabet

| 序号  | 图标  | LaTeX   | 英文     |
| :-: | :-: | :------ | :----- |
|  1  |  ℵ  | \aleph  | aleph  |
|  2  |  ℶ  | \beth   | beth   |
|  3  |  ℷ  | \gimel  | gimel  |
|  4  |  ℸ  | \daleth | daleth |

在Typora中使用markdown编写LaTex符号的效果：

π、ζ、π、ϱ、ℵ

源码为：

```
$\pi、\zeta、\pi、\varrho、\aleph$   # 一对$$
```

## 1.3 二元Binary
### 1.3.1 二元运算符 Binary operations

常用的二元运算符

|序号|图标|LaTeX|序号|图标|LaTeX|
|:-:|:-:|:--|:-:|:-:|:--|
|1|+|+|20|∙|\bullet|
|2|−|-|21|⊕|\oplus|
|3|×|\times|22|⊖|\ominus|
|4|÷\div÷|\div (在physics扩展开启状态下为)|23|⊙|\odot|
|5|±|\pm|24|⊘|\oslash|
|6|∓|\mp|25|⊗|\otimes|
|7|◃|\triangleleft|26|◯|\bigcirc|
|8|▹|\triangleright|27|⋄|\diamond|
|9|⋅|\cdot|28|⊎|\uplus|
|10|∖|\setminus|29|△|\bigtriangleup|
|11|⋆|\star|30|▽|\bigtriangledown|
|12|∗|\ast|31|⊲|\lhd|
|13|∪|\cup|32|⊳|\rhd|
|14|∩|\cap|33|⊴|\unlhd|
|15|⊔|\sqcup|34|⊵|\unrhd|
|16|⊓|\sqcap|35|⨿|\amalg|
|17|∨|\vee|36|≀|\wr|
|18|∧|\wedge|37|†|\dagger|
|19|∘|\circ|38|‡|\ddagger|

### 1.3.2 二元关系符 Binary relations

常见的二元关系符：

| 序号  |               图标               | LaTeX                      | 序号  | 图标  | LaTeX        |
| :-: | :----------------------------: | :------------------------- | :-: | :-: | :----------- |
|  1  |               =                | =                          | 49  |  ⪈  | \gneq        |
|  2  |               ≠                | \ne                        | 50  |  ≧  | \geqq        |
|  3  |               ≠                | \neq                       | 51  |  ≱  | \ngeq        |
|  4  |               ≡                | \equiv                     | 52  |  ≱  | \ngeqq       |
|  5  |               ≢                | \not\equiv                 | 53  |  ≩  | \gneqq       |
|  6  |               ≐                | \doteq                     | 54  |  ≩  | \gvertneqq   |
|  7  |               ≑                | \doteqdot                  | 55  |  ≶  | \lessgtr     |
|  8  | \stackrel{\mathrm{dff}}{=}=dff | \stackrel{\mathrm{dff}}{=} | 56  |  ⋚  | \lesseqgtr   |
|  9  |               :=               | :=                         | 57  |  ⪋  | \lesseqqgtr  |
| 10  |               ∼                | \sim                       | 58  |  ≷  | \gtrless     |
| 11  |               ≁                | \nsim                      | 59  |  ⋛  | \gtreqless   |
| 12  |               ∽                | \backsim                   | 60  |  ⪌  | \gtreqqless  |
| 13  |               ∼                | \thicksim                  | 61  |  ⩽  | \leqslant    |
| 14  |               ≃                | \simeq                     | 62  |  ⪇  | \nleqslant   |
| 15  |               ⋍                | \backsimeq                 | 63  |  ⪕  | \eqslantless |
| 16  |               ≂                | \eqsim                     | 64  |  ⩾  | \geqslant    |
| 17  |               ≅                | \cong                      | 65  |  ⪈  | \ngeqslant   |
| 18  |               ≇                | \ncong                     | 66  |  ⪖  | \eqslantgtr  |
| 19  |               ≈                | \approx                    | 67  |  ≲  | \lesssim     |
| 20  |               ≈                | \thickapprox               | 68  |  ⋦  | \lnsim       |
| 21  |               ≊                | \approxeq                  | 69  |  ⪅  | \lessapprox  |
| 22  |               ≍                | \asymp                     | 70  |  ⪉  | \lnapprox    |
| 23  |               ∝                | \propto                    | 71  |  ≳  | \gtrsim      |
| 24  |               ∝                | \varpropto                 | 72  |  ⋧  | \gnsim       |
| 25  |            << /td>             | <                          | 73  |  ⪆  | \gtrapprox   |
| 26  |               ≮                | \nless                     | 74  |  ⪊  | \gnapprox    |
| 27  |               ≪                | \ll                        | 75  |  ≺  | \prec        |
| 28  |               ≪̸               | \not\ll                    | 76  |  ⊀  | \nprec       |
| 29  |               ⋘                | \lll                       | 77  |  ⪯  | \preceq      |
| 30  |               ⋘̸               | \not\lll                   | 78  |  ⋠  | \npreceq     |
| 31  |               ⋖                | \lessdot                   | 79  |  ⪵  | \precneqq    |
| 32  |               80               |                            |     |  ≻  | \succ        |
| 33  |               ≯                | \ngtr                      | 81  |  ⊁  | \nsucc       |
| 34  |               ≫                | \gg                        | 82  |  ⪰  | \succeq      |
| 35  |               ≫̸               | \not\gg                    | 83  |  ⋡  | \nsucceq     |
| 36  |               ⋙                | \ggg                       | 84  |  ⪶  | \succneqq    |
| 37  |               ⋙̸               | \not\ggg                   | 85  |  ≼  | \preccurlyeq |
| 38  |               ⋗                | \gtrdot                    | 86  |  ⋞  | \curlyeqprec |
| 39  |               ≤                | \le                        | 87  |  ≽  | \succcurlyeq |
| 40  |               ≤                | \leq                       | 88  |  ⋟  | \curlyeqsucc |
| 41  |               ⪇                | \lneq                      | 89  |  ≾  | \precsim     |
| 42  |               ≦                | \leqq                      | 90  |  ⋨  | \precnsim    |
| 43  |               ≰                | \nleq                      | 91  |  ⪷  | \precapprox  |
| 44  |               ≰                | \nleqq                     | 92  |  ⪹  | \precnapprox |
| 45  |               ≨                | \lneqq                     | 93  |  ≿  | \succsim     |
| 46  |               ≨                | \lvertneqq                 | 94  |  ⋩  | \succnsim    |
| 47  |               ≥                | \ge                        | 95  |  ⪸  | \succapprox  |
| 48  |               ≥                | \geq                       | 96  |  ⪺  | \succnapprox |

## 1.4 符号Symbols
### 1.4.1 几何关系符号Geometric symbols

常见的几何关系符号：

|序号|图标|LaTeX|序号|图标|LaTeX|
|:-:|:-:|:--|:-:|:-:|:--|
|1|∥|\parallel|14|◊|\lozenge|
|2|∦|\nparallel|15|⧫|\blacklozenge|
|3|∥|\shortparallel|16|★|\bigstar|
|4|∦|\nshortparallel|17|◯|\bigcirc|
|5|⊥|\perp|18|△|\triangle|
|6|∠|\angle|19|△|\bigtriangleup|
|7|∢|\sphericalangle|20|▽|\bigtriangledown|
|8|∡|\measuredangle|21|△|\vartriangle|
|9|45∘|45^\circ|22|▽|\triangledown|
|10|◻|\Box|23|▴|\blacktriangle|
|11|◼|\blacksquare|24|▾|\blacktriangledown|
|12|⋄|\diamond|25|◂|\blacktriangleleft|
|13|◊|\Diamond \lozenge|26|▸|\blacktriangleright|

### 1.4.2 逻辑符号 Logic symbols

常用的逻辑关系运算符：

|序号|图标|LaTeX|序号|图标|LaTeX|
|:-:|:-:|:--|:-:|:-:|:--|
|1|∀|\forall|20|¬|\neg|
|2|∃|\exists|21|R̸|\not\operatorname{R}|
|3|∄|\nexists|22|⊥|\bot|
|4|∴|\therefore|23|⊤|\top|
|5|∵|\because|24|⊢|\vdash|
|6|&|\And|25|⊣|\dashv|
|7|∨|\lor|26|⊨|\vDash|
|8|∨|\vee|27|⊩|\Vdash|
|9|⋎|\curlyvee|28|⊨|\models|
|10|⋁|\bigvee|29|⊪|\Vvdash|
|11|∧|\land|30|⊬|\nvdash|
|12|∧|\wedge|31|⊮|\nVdash|
|13|⋏|\curlywedge|32|⊭|\nvDash|
|14|⋀|\bigwedge|33|⊯|\nVDash|
|15|q¯|\bar{q}|34|⌜|\ulcorner|
|16|abc¯|\bar{abc}|35|⌝|\urcorner|
|17|q―|\overline{q}|36|⌞|\llcorner|
|18|abc―|\overline{abc}|37|⌟|\lrcorner|
|19|¬|\lnot||||

### 1.4.3 集合Sets

集合相关的符号：空集、并集、子集、包含与被包含、存在于等各种符号

|序号|图标|LaTeX|序号|图标|LaTeX|
|:-:|:-:|:--|:-:|:-:|:--|
|1|{}|{}|23|⊏|\sqsubset|
|2|∅|\emptyset|24|⊃|\supset|
|3|∅|\varnothing|25|⋑|\Supset|
|4|∈|\in|26|⊐|\sqsupset|
|5|∉|\notin|27|⊆|\subseteq|
|6|∋|\ni|28|⊈|\nsubseteq|
|7|∩|\cap|29|⊊|\subsetneq|
|8|⋒|\Cap|30|⊊|\varsubsetneq|
|9|⊓|\sqcap|31|⊑|\sqsubseteq|
|10|⋂|\bigcap|32|⊇|\supseteq|
|11|∪|\cup|33|⊉|\nsupseteq|
|12|⋓|\Cup|34|⊋|\supsetneq|
|13|⊔|\sqcup|35|⊋|\varsupsetneq|
|14|⋃|\bigcup|36|⊒|\sqsupseteq|
|15|⨆|\bigsqcup|37|⫅|\subseteqq|
|16|⊎|\uplus|38|⊈|\nsubseteqq|
|17|⨄|\biguplus|39|⫋|\subsetneqq|
|18|∖|\setminus|40|⫋|\varsubsetneqq|
|19|∖|\smallsetminus|41|⫆|\supseteqq|
|20|×|\times|42|⊉|\nsupseteqq|
|21|⊂|\subset|43|⫌|\supsetneqq|
|22|⋐|\Subset|44|⫌|\varsupsetneqq|

### 1.4.4 箭头Arrows

和箭头相关的各种符号：

|序号|图标|LaTeX|序号|图标|LaTeX|
|:-:|:-:|:--|:-:|:-:|:--|
|1|⇛|\Rrightarrow|36|⟼|\longmapsto|
|2|⇚|\Lleftarrow|37|⇀|\rightharpoonup|
|3|⇒|\Rightarrow|38|⇁|\rightharpoondown|
|4|⇏|\nRightarrow|39|↼|\leftharpoonup|
|5|⟹|\Longrightarrow|40|↽|\leftharpoondown|
|6|⟹|\implies|41|↿|\upharpoonleft|
|7|⇐|\Leftarrow|42|↾|\upharpoonright|
|8|⇍|\nLeftarrow|43|⇃|\downharpoonleft|
|9|⟸|\Longleftarrow|44|⇂|\downharpoonright|
|10|⇔|\Leftrightarrow|45|⇌|\rightleftharpoons|
|11|⇎|\nLeftrightarrow|46|⇋|\leftrightharpoons|
|12|⟺|\Longleftrightarrow|47|↶|\curvearrowleft|
|13|⟺|\iff|48|↺|\circlearrowleft|
|14|⇑|\Uparrow|49|↰|\Lsh|
|15|⇓|\Downarrow|50|⇈|\upuparrows|
|16|⇕|\Updownarrow|51|⇉|\rightrightarrows|
|17|→|\rightarrow|52|⇄|\rightleftarrows|
|18|→|\to|53|↣|\rightarrowtail|
|19|↛|\nrightarrow|54|↬|\looparrowright|
|20|⟶|\longrightarrow|55|↷|\curvearrowright|
|21|←|\leftarrow|56|↻|\circlearrowright|
|22|←|\gets|57|↱|\Rsh|
|23|↚|\nleftarrow|58|⇊|\downdownarrows|
|24|⟵|\longleftarrow|59|⇇|\leftleftarrows|
|25|↔|\leftrightarrow|60|⇆|\leftrightarrows|
|26|↮|\nleftrightarrow|61|↢|\leftarrowtail|
|27|⟷|\longleftrightarrow|62|↫|\looparrowleft|
|28|↑|\uparrow|63|↪|\hookrightarrow|
|29|↓|\downarrow|64|↩|\hookleftarrow|
|30|↕|\updownarrow|65|⊸|\multimap|
|31|↗|\nearrow|66|↭|\leftrightsquigarrow|
|32|↙|\swarrow|67|⇝|\rightsquigarrow|
|33|↖|\nwarrow|68|↠|\twoheadrightarrow|
|34|↘|\searrow|69|↞|\twoheadleftarrow|
|35|↦|\mapsto|

### 1.4.5 特殊关系符号

其他少见的特殊符号：

|序号|图标|LaTeX|序号|图标|LaTeX|
|:-:|:-:|:--|:-:|:-:|:--|
|1|∞|\infty|33|♭|\flat|
|2|ℵ|\aleph|34|♮|\natural|
|3|∁|\complement|35|♯|\sharp|
|4|∍|\backepsilon|36|╱|\diagup|
|5|ð|\eth|37|╲|\diagdown|
|6|Ⅎ|\Finv|38|⋅|\centerdot|
|7|ℏ|\hbar|39|⋉|\ltimes|
|8|Im|\Im|40|⋊|\rtimes|
|9|ı|\imath|41|⋋|\leftthreetimes|
|10|ȷ|\jmath|42|⋌|\rightthreetimes|
|11|𝕜k|\Bbbk|43|≖|\eqcirc|
|12|ℓ|\ell|44|≗|\circeq|
|13|℧|\mho|45|≜|\triangleq|
|14|℘|\wp|46|≏|\bumpeq|
|15|Re|\Re|47|≎|\Bumpeq|
|16|Ⓢ|\circledS|48|≑|\doteqdot|
|17|⨿|\amalg|49|≓|\risingdotseq|
|18|%|%|50|≒|\fallingdotseq|
|19|†|\dagger|51|⊺|\intercal|
|20|‡|\ddagger|52|⊼|\barwedge|
|21|…|\ldots|53|⊻|\veebar|
|22|⋯|\cdots|54|⩞|\doublebarwedge|
|23|⌣\smile⌣|\smile|55|≬|\between|
|24|⌢\frown⌢|\frown|56|⋔|\pitchfork|
|25|≀|\wr|57|⊲|\vartriangleleft|
|26|◃|\triangleleft|58|⋪|\ntriangleleft|
|27|▹|\triangleright|59|⊳|\vartriangleright|
|28|♢|\diamondsuit|60|⋫|\ntriangleright|
|29|♡|\heartsuit|61|⊴|\trianglelefteq|
|30|♣|\clubsuit|62|⋬|\ntrianglelefteq|
|31|♠|\spadesuit|63|⊵|\trianglerighteq|
|32|⅁|\Game|64|⋭|\ntrianglerighteq|

## 1.5 数学Mathematics

下面介绍和数学运算相关的符号：

### 1.5.1 分数Fractions

|编号|类型|样式|LaTeX|
|---|:--|:--|:--|
|1|分数 Fractions|24x=0.5xor24x=0.5x\frac{2}{4}x=0.5x or {2 \over 4}x=0.5x42​x=0.5xor42​x=0.5x|\frac{2}{4}x=0.5x or {2 \over 4}x=0.5x|
|2|小型分数 Small fractions (force \textstyle)|24x=0.5x\tfrac{2}{4}x = 0.5x42​x=0.5x|\tfrac{2}{4}x = 0.5x|
|3|大型分数（不嵌套） Large (normal) fractions (force \displaystyle)|24=0.52c+2d+24=a\dfrac{2}{4} = 0.5 \qquad \dfrac{2}{c + \dfrac{2}{d + \dfrac{2}{4}}} = a42​=0.5c+d+42​2​2​=a|\dfrac{2}{4} = 0.5 \qquad \dfrac{2}{c + \dfrac{2}{d + \dfrac{2}{4}}} = a|
|4|大型分数（嵌套） Large (nested) fractions|2c+2d+24=a\cfrac{2}{c + \cfrac{2}{d + \cfrac{2}{4}}} = ac+d+42​2​2​=a|\cfrac{2}{c + \cfrac{2}{d + \cfrac{2}{4}}} = a|
|5|约分线的使用 Cancellations in fractions|x1+yy=x2\cfrac{x}{1 + \cfrac{\cancel{y}}{\cancel{y}}} = \cfrac{x}{2}1+y|

### 1.5.2 标准数值函数 Standard numerical functions

|编号|样式|LaTeX|
|---|:--|:--|
|1|exp⁡ab=ab,exp⁡b=eb,10m\exp_a b = a^b, \exp b = e^b, 10^mexpa​b=ab,expb=eb,10m|\exp_a b = a^b, \exp b = e^b, 10^m|
|2|ln⁡c,lg⁡d=log⁡e,log⁡10f\ln c, \lg d = \log e, \log_{10} flnc,lgd=loge,log10​f|\ln c, \lg d = \log e, \log_{10} f|
|3|sin⁡a,cos⁡b,tan⁡c,cot⁡d,sec⁡e,csc⁡f\sin a, \cos b, \tan c, \cot d, \sec e, \csc fsina,cosb,tanc,cotd,sece,cscf|\sin a, \cos b, \tan c, \cot d, \sec e, \csc f|
|4|arcsin⁡a,arccos⁡b,arctan⁡c\arcsin a, \arccos b, \arctan carcsina,arccosb,arctanc|\arcsin a, \arccos b, \arctan c|
|5|arccot⁡d,arcsec⁡e,arccsc⁡f\operatorname{arccot} d, \operatorname{arcsec} e, \operatorname{arccsc} farccotd,arcsece,arccscf|\operatorname{arccot} d, \operatorname{arcsec} e, \operatorname{arccsc} f|
|6|sinh⁡a,cosh⁡b,tanh⁡c,coth⁡d\sinh a, \cosh b, \tanh c, \coth dsinha,coshb,tanhc,cothd|\sinh a, \cosh b, \tanh c, \coth d|
|7|sh⁡k,ch⁡l,th⁡m,coth⁡n\operatorname{sh}k, \operatorname{ch}l, \operatorname{th}m, \operatorname{coth}nshk,chl,thm,cothn|\operatorname{sh}k, \operatorname{ch}l, \operatorname{th}m, \operatorname{coth}n|
|8|argsh⁡o,argch⁡p,argth⁡q\operatorname{argsh}o, \operatorname{argch}p, \operatorname{argth}qargsho,argchp,argthq|\operatorname{argsh}o, \operatorname{argch}p, \operatorname{argth}q|
|9|sgn⁡r,∣s∣\operatorname{sgn}r, \left\vert s \right\vertsgnr,∣s∣|\operatorname{sgn}r, \left\vert s \right\vert|
|10|min⁡(x,y),max⁡(x,y)\min(x,y), \max(x,y)min(x,y),max(x,y)|\min(x,y), \max(x,y)|

### 1.5.3 根式符号 Radicals

![image.png](https://aquazone.oss-cn-guangzhou.aliyuncs.com/20240302211830.png)

### 1.5.4 微分和导数 Differentials and derivatives

|编号|样式|LaTeX|
|---|:--|:--|
|1|dt,dt,∂t,∇ψ|dt, \mathrm{d}t, \partial t, \nabla\psi|
|2|dy/dx,dy/dx,dydx,dydx,∂2∂x1∂x2ydy/dx,dy/dx,dxdy​,dxdy​,∂x1​∂x2​∂2​y|dy/dx, \mathrm{d}y/\mathrm{d}x, \frac{dy}{dx}, \frac{\mathrm{d}y}{\mathrm{d}x}, \frac{\partial^2}{\partial x_1\partial x_2}y|
|3|′,‵,f′,f′,f′′,f(3),y˙,y¨′,‵,f′,f′,f′′,f(3),y˙​,y¨​|\prime, \backprime, f^\prime, f', f'', f^{(3)}, \dot y, \ddot y|
  
### 1.5.5 同余与模算术 Modular arithmetic

|编号|样式|LaTeX|
|---|:--|:--|
|1|sk≡0(modm)s_k \equiv 0 \pmod{m}sk​≡0(modm)|s_k \equiv 0 \pmod{m}|
|2|a  ba \bmod bamodb|a \bmod b|
|3|gcd⁡(m,n),lcm⁡(m,n)\gcd(m, n), \operatorname{lcm}(m, n)gcd(m,n),lcm(m,n)|\gcd(m, n), \operatorname{lcm}(m, n)|
|4|∣,∤,∣,∤\mid, \nmid, \shortmid, \nshortmid∣,∤,∣,|\mid, \nmid, \shortmid, \nshortmid|

### 1.5.6 极限 Limits

|编号|样式|LaTeX|
|---|:--|:--|
|1|lim⁡n→∞xn\lim_{n \to \infty}x_nlimn→∞​xn​|\lim_{n \to \infty}x_n|
|2|lim⁡n→∞xn\textstyle \lim_{n \to \infty}x_nlimn→∞​xn​|\textstyle \lim_{n \to \infty}x_n|

### 1.5.7 界限与投影 Bounds and Projections

|编号|样式|LaTeX|
|---|:--|:--|
|1|min⁡x,max⁡y,inf⁡s,sup⁡t\min x, \max y, \inf s, \sup tminx,maxy,infs,supt|\min x, \max y, \inf s, \sup t|
|2|lim⁡u,lim inf⁡v,lim sup⁡w\lim u, \liminf v, \limsup wlimu,liminfv,limsupw|\lim u, \liminf v, \limsup w|
|3|dim⁡p,deg⁡q,det⁡m,ker⁡ϕ\dim p, \deg q, \det m, \ker\phidimp,degq,detm,kerϕ|\dim p, \deg q, \det m, \ker\phi|
|4|Pr⁡j,hom⁡l,∥z∥,arg⁡z\Pr j, \hom l, \lVert z \rVert, \arg zPrj,homl,∥z∥,argz|\Pr j, \hom l, \lVert z \rVert, \arg z|

### 1.5.8 积分 Integral

|编号|样式|LaTeX|
|---|:--|:--|
|1|∫13e3/xx2 dx\int\limits_{1}^{3}\frac{e^3/x}{x^2}\, dx1∫3​x2e3/x​dx|\int\limits_{1}^{3}\frac{e^3/x}{x^2}, dx|
|2|∫13e3/xx2 dx\int_{1}^{3}\frac{e^3/x}{x^2}\, dx∫13​x2e3/x​dx|\int_{1}^{3}\frac{e^3/x}{x^2}, dx|
|3|∫−NNexdx\textstyle \int\limits_{-N}^{N} e^x dx−N∫N​exdx|\textstyle \int\limits_{-N}^{N} e^x dx|
|4|∫−NNexdx\textstyle \int_{-N}^{N} e^x dx∫−NN​exdx|\textstyle \int_{-N}^{N} e^x dx|
|5|∬Ddx dy\iint\limits_D dx\,dyD∬​dxdy|\iint\limits_D dx,dy|
|6|∭Edx dy dz\iiint\limits_E dx\,dy\,dzE∭​dxdydz|\iiint\limits_E dx,dy,dz|
|7|\iiiint\limits_F dx\,dy\,dz\,dt|\iiiint\limits_F dx,dy,dz,dt|
|8|∫(x,y)∈Cx3 dx+4y2 dy\int_{(x,y)\in C} x^3\, dx + 4y^2\, dy∫(x,y)∈C​x3dx+4y2dy|\int_{(x,y)\in C} x^3, dx + 4y^2, dy|
|9|∮(x,y)∈Cx3 dx+4y2 dy\oint_{(x,y)\in C} x^3\, dx + 4y^2\, dy∮(x,y)∈C​x3dx+4y2dy|\oint_{(x,y)\in C} x^3, dx + 4y^2, dy|
|10|\unicode8751\unicodex222FC\unicode{8751} \unicode{x222F}_C\unicode8751\unicodex222FC​|\unicode{8751} \unicode{x222F}_C|
|11|\unicode8752\unicodex2230C\unicode{8752} \unicode{x2230}_C\unicode8752\unicodex2230C​|\unicode{8752} \unicode{x2230}_C|
|12|\unicode8753\unicodex2231c\unicode{8753} \unicode{x2231}_c \unicode8753\unicodex2231c​|\unicode{8753} \unicode{x2231}_c|
|13|\unicode8754\unicodex2232c\unicode{8754} \unicode{x2232}_c \unicode8754\unicodex2232c​|\unicode{8754} \unicode{x2232}_c|
|14|\unicode8755\unicodex2233c\unicode{8755} \unicode{x2233}_c\unicode8755\unicodex2233c​|\unicode{8755} \unicode{x2233}_c|

最后三组图标的区别在于：![a978e474f6eb4e76830b3a5871498bf0~tplv-k3u1fbpfcp-zoom-in-crop-mark 1512 0 0 0.webp](https://aquazone.oss-cn-guangzhou.aliyuncs.com/a978e474f6eb4e76830b3a5871498bf0~tplv-k3u1fbpfcp-zoom-in-crop-mark%201512%200%200%200.webp)

### 1.5.9 其他运算符号operators

|编号|类别|样式|LaTeX|
|---|:--|:--|:--|
|1|求和 Summation|∑ab\sum_{a}^{b}∑ab​|\sum_{a}^{b}|
|2|求和 Summation|∑ab\textstyle \sum_{a}^{b}∑ab​|\textstyle \sum_{a}^{b}|
|3|连乘积 Product|∏ab\prod_{a}^{b}∏ab​|\prod_{a}^{b}|
|4|连乘积 Product|∏ab\textstyle \prod_{a}^{b}∏ab​|\textstyle \prod_{a}^{b}|
|5|余积 Coproduct|∐ab\coprod_{a}^{b}∐ab​|\coprod_{a}^{b}|
|6|余积 Coproduct|∐ab\textstyle \coprod_{a}^{b}∐ab​|\textstyle \coprod_{a}^{b}|
|7|并集 Union|⋃ab\bigcup_{a}^{b}⋃ab​|\bigcup_{a}^{b}|
|8|并集 Union|⋃ab\textstyle \bigcup_{a}^{b}⋃ab​|\textstyle \bigcup_{a}^{b}|
|9|交集 Intersection|⋂ab\bigcap_{a}^{b}⋂ab​|\bigcap_{a}^{b}|
|10|交集 Intersection|⋂ab\textstyle \bigcap_{a}^{b}⋂ab​|\textstyle \bigcap_{a}^{b}|
|11|析取 Disjunction|⋁ab\bigvee_{a}^{b}⋁ab​|\bigvee_{a}^{b}|
|12|析取 Disjunction|⋁ab\textstyle \bigvee_{a}^{b}⋁ab​|\textstyle \bigvee_{a}^{b}|
|13|合取 Conjunction|⋀ab\bigwedge_{a}^{b}⋀ab​|\bigwedge_{a}^{b}|
|14|合取 Conjunction|⋀ab\textstyle \bigwedge_{a}^{b}⋀ab​|\textstyle \bigwedge_{a}^{b}|

### 1.5.10 上下标Sub & Super

|编号|类型|样式|代码|
|---|:--|---|---|
|1|上标 Superscript|a2,ax+3a^2, a^{x+3}a2,ax+3|a^2, a^{x+3}|
|2|下标 Subscript|a2a_2a2​|a_2|
|3|组合1 Grouping|1030a2+210^{30} a^{2+2}1030a2+2|10^{30} a^{2+2}|
|4|组合2 Grouping|ai,jbf′a_{i,j} b_{f'}ai,j​bf′​|a_{i,j} b_{f'}|
|5|上下标混合1 Combining sub & super|x23x_2^3x23​|x_2^3|
|6|上下标混合2 Combining sub & super|x23{x_2}^3x2​3|{x_2}^3|
|7|上标的上标 Super super|1010810^{10^{8}}10108|10^{10^{8}}|
|8|混合标识1 Preceding and/or additional sub & super|\sideset1234Xab\sideset{_1^2}{_3^4}X_a^b\sideset12​34​Xab​|\sideset{_1^2}{_3^4}X_a^b|
|9|混合标识2 Preceding and/or additional sub & super|12!Ω34{}_1^2!\Omega_3^412​!Ω34​|{}_1^2!\Omega_3^4|
|10|顶标底标1 Stacking|\overset{\alpha}{\omega}ωα|\overset{\alpha}{\omega}|
|11|顶标底标2 Stacking|\underset{\alpha}{\omega}αω​|\underset{\alpha}{\omega}|
|12|顶标底标3 Stacking|\overset{\alpha}{\underset{\gamma}{\omega}}γω​α​|\overset{\alpha}{\underset{\gamma}{\omega}}|
|13|顶标底标4 Stacking|\stackrel{\alpha}{\omega}ωα|\stackrel{\alpha}{\omega}|
|14|导数1 Derivatives|x′,y′′,f′,f′′x', y'', f', f''x′,y′′,f′,f′′|x', y'', f', f''|
|15|导数2 Derivatives|x′,y′′x^\prime, y^{\prime\prime}x′,y′′|x^\prime, y^{\prime\prime}|
|16|导数 Derivative dots|x˙,x¨\dot{x}, \ddot{x}x˙,x¨|\dot{x}, \ddot{x}|
|17|下划线、上划线与向量1 Underlines, overlines, vectors|a^ bˉ c⃗\hat a \ \bar b \ \vec ca^ bˉ c|

### 1.5.11 矩阵与多行列式 Matrices & Multilines？

## 1.6 格式Format

### 1.6.1 括号Brackets？

### 1.6.2 空格与换行

关于空格的使用可以参考下表。

|编号|样式|LaTeX|中文说明英文说明|
|:-:|:-:|:--|:--|
|1|aba \qquad bab|a \qquad b|双空格|
|2|aba \quad bab|a \quad b|单空格|
|3|a ba\ ba b|a\ b|字符空格|
|4|a ba \text{ } ba b|a \text{ } b|文本模式中的字符空格|
|5|a  ba\;bab|a;b|大空格|
|6|a ba\,bab|a,b|小空格|
|7|ababab|ab|极小空格(用于乘因子)|
|8|aba bab|a b|极小空格(用于区分其它语法)|
|9|ab\mathit{ab}ab|\mathit{ab}|没有空格(用于多字母变量)|
|10|a ⁣ba\!bab|a\!b|负空格|

关于换行，使用一对`\\`进行强制换行，看一个案例：

f(x,y)=a2+b2+2ab=(a+b)2f(x,y) = a^2+b^2+2ab \\ =(a+b)^2f(x,y)=a2+b2+2ab=(a+b)2

实际源码为：

```
$$f(x,y) = a^2+b^2+2ab \\ =(a+b)^2$$
```
