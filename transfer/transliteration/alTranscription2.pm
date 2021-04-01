package alTranscription;
use utf8;
use strict;
use alVariables;
use Exporter;
#use Locale;
use locale;
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
						      "Ə" => "E",
						      "E" => "Ë",
						      "Q" => "G",
						      "G" => "DY ",
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
						      "Ә" => "E",
						      "Б" => "B",
						      "В" => "V",
						      "Г" => "G",
						      "Ғ" => "Ğ",
						      "Д" => "D",
						      "Е" => "YI ",
						      "Ё" => "YO ",
						      "Ж" => "J",
						      "З" => "Z",
						      "И" => "İY",
						      "І" => "İ",
						      "Й" => "Y",
						      "К" => "K",
						      "Қ" => "Q",
						      "Л" => "L",
						      "М" => "M",
						      "Н" => "N",
						      "Ң" => "NG ",
						      "О" => "O",
						      "Ө" => "Ö",
						      "П" => "P",
						      "Р" => "R",
						      "С" => "S",
						      "Т" => "T",
						      "У" => "W",
						      "Ү" => "Ü",
						      "Ұ" => "U",
						      "Ф" => "F",
						      "Х" => "X",
						      "Һ" => "H",
						      "Ц" => "TS ",
						      "Ч" => "Ç",
						      "Ш" => "Ş",
						      "Щ" => "ÇŞ ",
						      "Ъ" => "",
						      "Ы" => "I",
						      "Ь" => "",
						      "Э" => "Ë",
						      "Ю" => "YO ",
						      "Я" => "YA ",
						     },
					     'ALL' => {
						       "،" => ",",
						       },
					    }
				    );

sub al_transcribe {
  my ($s,$slang,$tlang) = @_;
  $s =~ s/ / /g;
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
  $s =~ s/([A-ZÇŞËÖÜĞİ][A-ZÇŞËÖÜĞİ][A-ZÇŞËÖÜĞİ]) /$1/g;
  $s =~ s/([A-ZÇŞËÖÜĞİ][A-ZÇŞËÖÜĞİ]) ([A-ZÇŞËÖÜĞİ])/$1$2/g;
  $s =~ s/(.) /lc_tr($1)/ge;
  return $s;
}

sub lc_tr {
  my $c = shift;
  if ($c eq "Ç") {return "ç"}
  if ($c eq "Ş") {return "ş"}
  if ($c eq "Ğ") {return "ğ"}
  if ($c eq "I") {return "ı"}
  if ($c eq "İ") {return "i"}
  return lc($c);
}

1;


