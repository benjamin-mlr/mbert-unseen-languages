package alTranscription;
use utf8;
use strict;
use alVariables;
use Exporter;
use Locale;
our @ISA = 'Exporter';
our @EXPORT = qw(&al_transcribe);

my %al_transcription_hash = (
				    'tr' => {
					     'ug' => {
						      "ئا" => "a",
						      "ئە" => "e",
						      "ا" => "a",
						      "ە" => "e",
						      "ب" => "b",
						      "پ" => "p",
						      "ت" => "t",
						      "ج" => "c",
						      "چ" => "ç",
						      "خ" => "x",
						      "د" => "d",
						      "ر" => "r",
						      "ز" => "z",
						      "ژ" => "j",
						      "س" => "s",
						      "ش" => "ş",
						      "غ" => "ğ",
						      "ف" => "f",
						      "ق" => "q",
						      "ك" => "k",
						      "گ" => "g",
						      "ڭ" => "ng",
						      "ل" => "l",
						      "م" => "m",
						      "ن" => "n",
						      "ھ" => "h",
						      "ح" => "h",
						      "ئو" => "o",
						      "ئۇ" => "u",
						      "و" => "o",
						      "ۇ" => "u",
						      "ئۆ" => "ö",
						      "ئۈ" => "ü",
						      "ۆ" => "ö",
						      "ۈ" => "ü",
						      "ۋ" => "v",
						      "ئې" => "ë",
						      "ئى" => "i",
						      "ي" => "y",
						      "ې" => "ë",
						      "ى" => "i",
						      "ء" => "",
						      "ع" => "kh",
						     },
					     'az' => {
						      "ə" => "e",
						      "e" => "ë",
						      "q" => "g",
						      "g" => "dy",
						     },
					     'kk' => {
						      "ә" => "e",
						      "б" => "b",
						      "в" => "v",
						      "г" => "g",
						      "ғ" => "ğ",
						      "д" => "d",
						      "е" => "yi",
						      "ё" => "yo",
						      "ж" => "j",
						      "з" => "z",
						      "и" => "iy",
						      "і" => "i",
						      "й" => "y",
						      "к" => "k",
						      "қ" => "q",
						      "л" => "l",
						      "м" => "m",
						      "н" => "n",
						      "ң" => "ng",
						      "о" => "o",
						      "ө" => "ö",
						      "п" => "p",
						      "р" => "r",
						      "с" => "s",
						      "т" => "t",
						      "у" => "w",
						      "ү" => "ü",
						      "ұ" => "u",
						      "ф" => "f",
						      "х" => "x",
						      "һ" => "h",
						      "ц" => "ts",
						      "ч" => "ç",
						      "ш" => "ş",
						      "щ" => "çş",
						      "ъ" => "",
						      "ы" => "ı",
						      "ь" => "",
						      "э" => "ë",
						      "ю" => "yo",
						      "я" => "ya",
						     },
					     'ALL' => {
						       "،" => ",",
						       },
					    }
				    );

sub al_transcribe {
  my ($s,$slang,$tlang) = @_;
  die "ERROR: transcription from language $slang to language $tlang has not been implemented yet" unless defined $al_transcription_hash{$tlang}{$slang};
  if (defined($al_transcription_hash{$tlang}{ALL})) {
    for my $in (sort {length($b) <=> length($a)} keys %{$al_transcription_hash{$tlang}{ALL}}) {
      my $out = $al_transcription_hash{$tlang}{ALL}{$in};
      $s =~ s/$in/$out/g;
    }
  }
  for my $in (sort {length($b) <=> length($a)} keys %{$al_transcription_hash{$tlang}{$slang}}) {
    my $out = $al_transcription_hash{$tlang}{$slang}{$in};
    $s =~ s/$in/$out/g;
  }
  return $s;
}

1;

