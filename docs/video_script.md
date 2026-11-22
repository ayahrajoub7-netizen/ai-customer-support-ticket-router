# 2-minute demo video – script

Record your screen with narration (OBS Studio, the Windows Game Bar `Win+Alt+R`, or QuickTime on Mac).
Before recording: run `python app.py`, open http://127.0.0.1:7860, and have the slides open on the
architecture slide in another window. Upload to YouTube as **Unlisted** (or Google Drive "anyone with the link")
and open the link in a private/incognito window to check it works.

---

**0:00 – 0:15 · The problem** (show the app's start page)

> "A home appliance company gets thousands of support messages a day, and each one has to reach the right team
> instantly. My application reads a customer message, predicts its intent and routes it to a team, and sends
> anything it isn't sure about to a human agent."

**0:15 – 1:15 · Live demo** (type each message yourself, don't only click the examples)

1. Type `my dishwaser is not draining, model DW-6534` → press *Route message*.
   > "There's a spelling mistake and a model code. The regex step pulls out the model code DW-6534,
   > edit distance corrects 'dishwaser' to 'dishwasher', and the quantized BERT-mini model predicts
   > dishwasher_issue with high confidence, so it goes to Appliance Technical Support."
2. Type `I lost my credit card yesterday, please block it`.
   > "A banking message goes to the Credit Card team. In the table below, you can see all five models side by side
   > with their latency. The INT8 model answers in a few milliseconds."
3. Type `Hello!`.
   > "Greetings don't need a team. An ELIZA-style regex rule answers them."
4. **Difficult case (out-of-scope):** type `who won the football match last night`.
   > "Our support doesn't handle this request. The model predicts out-of-scope, so the message goes to a
   > human agent instead of being misrouted."
5. **Difficult case (failure), if you have time:** type `what's the best way to learn to play the guitar`.
   > "Here's a failure. 'Play' and 'guitar' look like the play_music intent, so the model routes it with high
   > confidence even though it's out of scope. Confident mistakes like this are a limitation of our model."

**1:15 – 1:45 · Architecture and results** (switch to the slides)

> "Here's the architecture. Preprocessing uses regular expressions and edit distance. The main model is a BERT-mini
> distilled from a fine-tuned BERT-base and then quantized to INT8, and everything runs on the CPU.
> The results table compares all models on the same 5,520 test messages: the tf-idf baseline, my word2vec
> feed-forward network, BERT-base, and the distilled and quantized student, with accuracy, out-of-scope recall,
> size and latency."
> (Read out 2 numbers from YOUR table, e.g. the accuracy of BERT-base vs. the INT8 model, and how much smaller it is.)

**1:45 – 2:00 · Conclusion**

> "BERT is the most accurate, the tf-idf baseline is surprisingly strong and cheap, and distillation plus
> quantization give a model many times smaller and faster with only a small loss in accuracy. Next I would add
> Turkish messages with a multilingual model. Thank you."
