/* global document */
/* global jQuery */

(function ($) {

    'use strict';

    var loadYoutube = function() {
        $('.mace-youtube').each(function() {
            var $youtube = $(this);

            if($youtube.is('.mace-youtube-loaded')) {
                return;
            }

            var videoUrl = $youtube.attr('data-mace-video');
            var thumbUrl = $youtube.attr('data-mace-video-thumb');

            var image = $('<img />');
            image.attr('src', thumbUrl);

            // Lazy loading.
            image.attr({
                'width':        1600,
                'height':       900,
                'src':          'data:image/svg+xml;charset=utf-8,%3Csvg xmlns%3D\'http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg\' viewBox%3D\'0 0 1600 900\'%2F%3E',
                'data-src':     thumbUrl,
                'data-expand':  600
            });

            image.addClass('lazyload');

            image.on('load', function() {
                $youtube.append( image );
            }());

            // Click to play YT.
            $youtube.on('click', function() {
                $youtube.find('iframe').remove();

                var $iframe = $('<iframe>');

                $iframe.on('load', function() {
                    $youtube.trigger('maceIframeLoaded', [$iframe]);

                    // Play video.
                    $iframe[0].contentWindow.postMessage(JSON.stringify({
                            'event': 'command',
                            'func': 'playVideo',
                            'args': ''}),
                        '*');
                });

                $iframe.attr('frameborder', '0');
                $iframe.attr('allowfullscreen', '');
                $iframe.attr('src', videoUrl);

                $youtube.empty().append($iframe);
            });

            $youtube.addClass('mace-youtube-loaded');
        });
    };

    $(document).ready(function() {

        loadYoutube();

        $('body').on('maceLoadYoutube', loadYoutube);

    });

})(jQuery);

