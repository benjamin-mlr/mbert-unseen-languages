package alVariables;
use utf8;
use strict;
use Exporter;
our @ISA = 'Exporter';
our @EXPORT = qw($comment_content $alPunctuation $alAlphanum $alLowercase $alUppercase $alNonAlphanum);

our $comment_content = qr/(?:[^{}\\]|\\.)*/o;
our $alPunctuation = qr/\\?[!¡"'`´(),\-:;<=>?¿\#\%\&\*\+\.\/\=\[\\_\]\{\|\}«»˝។៕៖៘—‘’“”‥…॥।、。《》「」『』【】［］・！（）＋，〜－／：；＝？♪·;\t\x{05be}\x{05c0}\x{05c3}\x{05f3}\x{05f4}\x{061f}\x{2e2e}\x{060c}\x{06d4}]/o;
our $alAlphanum = qr/[0-9A-ZА-ЯЀ-ЏҊ-ӸͰ-ϿÀ-ɏa-zа-яѐ-џҋ-ӹａ-ｚＡ-Ｚ]/o;
our $alNonAlphanum = qr/[^0-9A-ZА-ЯЀ-ЏҊ-ӸͰ-ϿÀ-ɏa-zа-яѐ-џҋ-ӹａ-ｚＡ-Ｚ]/o;
our $alLowercase = qr/[a-zа-яѐ-џΐά-ώϐϑϕ-ϗϙϛϝϟϡϣϥϧϩϫϭϯϰϱϲϳϵ϶ϸϻϼáàâäąãăåćčçďéèêëęěğìíîĩĭïĺľłńñňòóôõöøŕřśšşťţùúûũüǔỳýŷÿźẑżž]/o;
our $alUppercase = qr/[A-ZА-ЯЀ-ЏΆ-ΏΑ-ΫϏϒ-ϔϘϚϜϞϠϢϤϦϨϪϬϮϴϷϹϺϽϾϿÁÀÂÄĄÃĂÅĆČÇĎÉÈÊËĘĚĞÌÍÎĨĬÏĹĽŁŃÑŇÒÓÔÕÖØŔŘŚŠŞŤŢÙÚÛŨÜǓỲÝŶŸŹẐŻŽ]/o;

1;

