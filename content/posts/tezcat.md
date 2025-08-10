---
date: '2025-08-10T13:54:00-04:00'
draft: false
title: 'tezcat - AI retrieval in Obsidian'
math: true
categories: ['machine learning', 'artificial intelligence', 'knowledge management', 'writing']
tags: ['embeddings', 'machine learning', 'remembrance agent', 'information retrieval']
---


Artificial intelligence (AI) products are now ubiquitous, but I think that few of them are able to hit on the right interface. Chat became the dominant functioning path too early, and now it is stapled onto everything, regardless of whether it is suitable for the medium of the information it is trying to channel.

When it comes to knowledge management, I personally do not like heavy usage of generative AI tools. For me the writing and the iterating is what makes the process worth doing, and from a [variety](https://www.nature.com/articles/s44222-025-00323-4) [of](https://www.paulgraham.com/writes.html) [works](https://en.wikipedia.org/wiki/Understanding_Media), we know that the act of writing contributes heavily to cementing new ideas in your mind. 

Some time ago, I ran into [this paper about remembrance agents](https://www.bradleyrhodes.com/Papers/remembrance.html), and it has just been fermenting as I turn it over. When dealing with notes, writing, knowledge management, the core problem [after creating information] is the retrieval of that information, and so an assist engineered to perform dynamic retrieval is a natural integration.

To that end, I built a plugin for Obsidian called [`tezcat`](https://github.com/mmargenot/tezcat) that uses text embeddings and search to do recall of fragments of notes that you've written in the past based on what you're writing right now. I'm currently dogfooding it to figure out what it might be missing, but on the whole I have found the persistent search to be consistently interesting, even if I only use the results to directly connect information via links occasionally.

It is **local-first** and FOSS, and should be easy to add to any existing vault.

![tezcat running on Edgar Allen Poe's collected works](tezcat/tezcat_search_2.gif)

Try it out and let me know what you think. If you like it enough, [feel free to contribute](https://github.com/mmargenot/tezcat/issues).
