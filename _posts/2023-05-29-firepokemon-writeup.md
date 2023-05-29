---
title: Sifting through a pcap file
published: true
description: This writeup details my process for completing a particularly tricky Capture the Flag (CTF) challenge that required parsing a pcapg file for a hidden flag. The pcapg file seemingly hinted at remote 
category: writeup
author: gary23w
featured_image: /assets/images/post1.png
category: 
    - CTF
tags:
    - CTF
    - forensics
---

# CTF Challenge Writeup: Sifting Through a pcapg File

## Introduction

This writeup details my process for completing a particularly tricky Capture the Flag (CTF) challenge that required parsing a pcapg file for a hidden flag. The pcapg file seemingly hinted at remote access to a vulnerable user. This writeup will explain the steps taken to uncover the flag, `UMDCTF{its_n0t_p1kachu!!}`.

## Initial Analysis

My journey began with the analysis of the pcapg file using the well-known network protocol analyzer, [Wireshark](https://www.wireshark.org/). My primary goal was to identify any suspicious or unusual network activity that may contain the flag.

## Exploiting FTP

Using Wireshark's hierarchy tool, I discovered that an FTP service was accessed. This piqued my interest and I decided to delve deeper into this. I used Wireshark's powerful filtering capabilities to whittle down to the exact packet where a transfer occurred.

## Extraction of Transferred Files

My next step involved exporting the objects transferred in the pcapg file. Wireshark's `Export Objects` tool is particularly useful for this task. From the transfer, I was able to extract several files - specifically, three images and a password-protected zip file.

## Hunting for the Password

The zip file was a point of intrigue but was locked behind a password, which was not known at that point. Hence, I took a step back and scoured the pcapg file for any credentials that could be useful. After some considerable effort, a set of credentials, a username and a password ("pika"), were found.

## Unveiling the Flag

The password "pika" successfully opened the zip file which contained a video. Inside this video was the hidden flag, `UMDCTF{its_n0t_p1kachu!!}`.

## Conclusion

Solving this CTF challenge was a rewarding experience that required detailed analysis of network activity, familiarity with a range of tools, and a healthy dose of perseverance. The thrill of the hunt and the satisfaction of the find is what makes Capture the Flag challenges so exciting and compelling.
