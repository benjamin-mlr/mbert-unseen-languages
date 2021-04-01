#!/usr/bin/perl

use utf8;
binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";
use strict;

use alTranscription;

my ($slang, $tlang);

while (1) {
  $_ = shift;
  if (/^$/) {last}
  elsif (/^-f$/) {$slang = shift || die "Option '-l' must be followed by a language code"}
  elsif (/^-t$/) {$tlang = shift || die "Option '-l' must be followed by a language code"}
  else {die "Unknown option '$_'"}
}

die "Please specify a source language (-f <lang>) and a target language (-t <lang>)" unless $slang ne "" && $tlang ne "";

while (<>) {
  chomp;
  print al_transcribe($_,$slang,$tlang)."\n";
}
