import os
import shutil

import gradio as gr
import joblib
from preprocess import Preprocess
from sklearn.pipeline import make_pipeline

from segmenter import NgramSegmenter

model = "bin/bins.zip"
if not os.path.exists("bin/model.bin"):
    shutil.unpack_archive(model, "bin")

label_encoder = joblib.load("bin/le.bin")
tfidf_vectorizer = joblib.load("bin/tfidf.bin")
model = joblib.load("bin/model.bin")

pipeline = make_pipeline(Preprocess(), tfidf_vectorizer, model)

segmenter = NgramSegmenter(pipeline, label_encoder)


def segment(text: str, ngram: int = 6, k: int = 4):
    """
    Segment text for gradio.
    """
    text_array, _, spans = segmenter.extract(text, int(ngram), int(k))
    outs = []
    for span, class_ in spans:
        out = (" ".join(text_array[span[0] : span[1]]), class_)
        outs.append(out)
    return outs


langs = [
    "igbo",
    "hausa",
    "igede",
    "kanuri",
    "ibibio",
    "pidgin",
    "others",
    "igala",
    "fulfude-adamawa",
    "idoma",
    "tiv",
    "yoruba",
    "english",
    "ebira",
    "efik",
]
langs = "".join([f'<li style="display: inline;">{i}</li>' for i in langs])
description = f""" This app segments a piece of text by languages (Nigerian Languages). Supported languages are:
                  <ul>
                  {langs}
                  </ul>
              """

examples = [
    [
        "Omo I no know how e take be like this. Ist das der engel teufel. I don't know what is going on outside right now. 'ewn chi akanya we ojane ile i'. Ingila a Fabrairun 2017 daga nan ya maye gurbin Howard Webb a",
        4,
        6,
    ],
    [
        """yoyi da yawa sun yi barazanar daina buga gasar kasar, domin nuna bacin ransu kan hukunce-hukuncen da raflin ke yankewa na rashin adalci.
Clattenburg ya hura wasa 570 a Ingila da nahiyar Turai da duniya, tsakanin 2000 zuwa 2017.
Ya nuna kwarewarsa a wasa biyu da ya hura a 2016 a Champions League karawar karshe a Milan da gasar kofin nahiyar Turai tsakanin Portugal da Faransa. 
Mai shekara 47, ya bar hura wasannin gasar Ingila a Fabrairun 2017 daga nan ya maye gurbin Howard Webb.
Abo israe̩l ka ñwu jose̩f kakini, dagba u leku ke, abu kʼu fʼeju-we̩ li, kʼe̩ la deju-i. 
u do̩mo̩ agwoguyo̩ ef ile̩-wñ kʼi chʼugbo kʼone̩ adodo
jonatan kpai ahimaaz la jo̩ en-roge̩l anede ohi-wñ; oñ imo̩to̩ onobule̩ʼka te̩te̩chʼe̩ne̩ kʼi ajuche̩ rʼuma; taku ma le tekeka ñwu defid onu; todu ku ma ki we enyʼef ewo tode̩gba kʼamone̩ ali ma. 
Eyi bayi li a o fi ridi nyin, nipa ẹmi Farao bi ẹnyin o ti lọ nihin, bikoṣepe arakunrin abikẹhin nyin wá ihinyi.
Ẹ rán ẹnikan ninu nyin, ki o si mú arakunrin nyin wá, a o si pa nyin mọ́ ninu túbu, ki a le ridi ọ̀rọ nyin, bi otitọ wà ninu nyin: bikoṣe bẹ̃, ẹmi Farao bi amí ki ẹnyin iṣe.
O si kó gbogbo wọn pọ̀ sinu túbu ni ijọ́ mẹta.
ve hingir ikav i tesen.
Mban kpaa ka sha gbenda shon môm môm ne ve lu hôngor ayol a ve sha mnyam ma cien, ve lu vendan mbautahav, ve lu tuhwan mba ve engem la ye.
Kpa Mikael ortyom u Sha u tamen la, zum u a lu nôngon a diabolo, lu kpelan ikom i Mose la yô, a kera cihi a ôr un ijir i sha mtoho ga, kpa a kaa er: Ter A̱ tsaha u.
Ior mban gema mba tuhwan hanma kwagh u ve fe un ga yô, man akaa a ve kav a sha marami la er uzendenya mba fan kwagh ga nahan la, ka sha a je ve lu timin ye.
Kwagh á tser ve, gadia mba zenden sha gbenda u.
hī ka Ananíyāsi. Óndú ā da ɔ́ ipú ɔ̄na kahíníī, “Ananíyāsi.” Anɔ́ɔ ó tóohi kahíníī, “Ami yɔ̄ á, Óndú.” 
Óndú ā da ɔ́ kahíníī, “Cāŋjɛ ká ā nyɔ̄ ga ɔlɛ́ kú Ujúdasi nōo yɔ̄ úláyi nēé hī ka Okpaakpa ā, cɛ́ɛ́ ká ā dɔɔ̄kā ɔ̄cɛ éyi nēé hī ka Usɔ́lu, ɔ̄cɛ kú Utásɔsi. Ó yɔ̄ ī gbɔ̄ɔkɔ, 
anɔ́ɔ ipú ɔ̄na ó le ɔ̄cɛ éyi nēé hī ka Ananíyāsi ā má abɔ̄ ó bēhé gáā le abɔ̄ ce ɔ́ ɛyí, cɛ́ɛ́ kóō klla lɔfú mɛ́ɛbɛ kpɔ́ á.” 
Amáŋ Ananíyāsi tóohi kahíníī, “Ón
yị ịlẹ kị ma bwula ọngịrị nya ugbinyịrọ ọlam-ọlam ọlẹ, a ri anyị ịlam-ịlam nya Ohe Oluhye ka! Ma, anyị ịlẹ kị ma bwula ọ-họ jịra nya ọnụ ọmyịmyị nya Ohe Oluhye kpangga, Ohe Oluhye a kaa wa 
nya anyị ịlam-ịlam ị-ịlahị nya ịlọmwụ lẹ! 
Chajị, ọnụ ọmyịmyị ọlẹ ká Ohe Oluhye myị lụka ọọwa ha Ebiraham a lẹẹ lẹ: “Ụka kọ ka pwụ ihwọngkịla amachanyị ịịlẹ kpá, m ka tịrẹk
""",
        4,
        6,
    ],
]
article = '<div style="text-align: center; width: 100%">By \
            <a href="https://www.github.com/matt-wisdom">Matthew Wisdom</a>|\
            <a href="https://www.github.com/naija-segmenter">Project Github Page</a></div>'


text = gr.Textbox(label="Multilingual text sample.")
ngram = gr.Number(4, label="ngrams")
k = gr.Number(
    6,
    label="smoothening parameter (should be smaller for very short texts)."
)
out = gr.HighlightedText(label="Language segments")
app = gr.Interface(
    fn=segment,
    inputs=[text, ngram, k],
    outputs=out,
    capture_session=True,
    title="Naija Segmenter",
    description=description,
    examples=examples,
    allow_flagging=False,
    article=article,
    cache_examples=True,
)
app.launch()
