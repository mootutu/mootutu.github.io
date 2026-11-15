(function () {
  'use strict';

  /* Header border once the page scrolls */
  var header = document.getElementById('site-header');
  if (header) {
    var onScroll = function () {
      header.classList.toggle('is-scrolled', window.scrollY > 8);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  /* Mobile menu */
  var navToggle = document.getElementById('nav-toggle');
  var navList = document.getElementById('site-nav-list');
  if (navToggle && navList) {
    var setMenu = function (open) {
      navList.classList.toggle('is-open', open);
      navToggle.setAttribute('aria-expanded', String(open));
      navToggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    };
    navToggle.addEventListener('click', function () {
      setMenu(!navList.classList.contains('is-open'));
    });
    navList.addEventListener('click', function (e) {
      if (e.target.closest('a')) setMenu(false);
    });
    document.addEventListener('click', function (e) {
      if (!navList.contains(e.target) && !navToggle.contains(e.target)) setMenu(false);
    });
  }

  /* Homepage: highlight the nav item of the section in view */
  var navLinks = Array.prototype.slice.call(document.querySelectorAll('.site-nav__link[data-section]'));
  var spySections = navLinks
    .map(function (a) { return document.getElementById(a.getAttribute('data-section')); })
    .filter(Boolean);
  if (spySections.length) {
    var setCurrent = function (id) {
      navLinks.forEach(function (a) {
        if (a.getAttribute('data-section') === id) a.setAttribute('aria-current', id === 'top' ? 'page' : 'location');
        else a.removeAttribute('aria-current');
      });
    };
    var updateSpy = function () {
      var current = 'top';
      var line = window.innerHeight * 0.35;
      spySections.forEach(function (sec) {
        if (sec.getBoundingClientRect().top <= line) current = sec.id;
      });
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
        current = spySections[spySections.length - 1].id;
      }
      setCurrent(current);
    };
    updateSpy();
    window.addEventListener('scroll', updateSpy, { passive: true });
    window.addEventListener('resize', updateSpy);
  }

  /* Hidden entrance: tap the orange dot 5 times, or type "mootutu" */
  var secretDialog = document.getElementById('secret-dialog');
  var secretMark = document.querySelector('.wordmark__mark');
  if (secretDialog) {
    var openSecret = function () {
      if (!secretDialog.open) secretDialog.showModal();
    };
    closeOnBackdrop(secretDialog);
    var secretClose = document.getElementById('secret-close');
    if (secretClose) secretClose.addEventListener('click', function () { secretDialog.close(); });

    if (secretMark) {
      var taps = 0;
      var tapTimer = null;
      secretMark.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        secretMark.classList.remove('is-tapped');
        void secretMark.offsetWidth;
        secretMark.classList.add('is-tapped');
        taps += 1;
        clearTimeout(tapTimer);
        tapTimer = setTimeout(function () { taps = 0; }, 1500);
        if (taps >= 5) {
          taps = 0;
          openSecret();
        }
      });
    }

    var secretWord = 'mootutu';
    var typed = '';
    document.addEventListener('keydown', function (e) {
      var t = e.target;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      if (e.key.length !== 1) return;
      typed = (typed + e.key.toLowerCase()).slice(-secretWord.length);
      if (typed === secretWord) {
        typed = '';
        openSecret();
      }
    });
  }

  /* Scroll reveal */
  var revealTargets = document.querySelectorAll('[data-reveal]');
  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
    revealTargets.forEach(function (el) { observer.observe(el); });
  } else {
    revealTargets.forEach(function (el) { el.classList.add('is-visible'); });
  }

  /* Close a <dialog> when its backdrop is clicked */
  function closeOnBackdrop(dialog) {
    dialog.addEventListener('click', function (e) {
      if (e.target === dialog) dialog.close();
    });
  }

  /* Search palette */
  var searchDialog = document.getElementById('search-dialog');
  var searchOpen = document.getElementById('search-open');
  var searchInput = document.getElementById('search-input');
  var searchResults = document.getElementById('search-results');
  var searchData = null;

  function loadSearchData() {
    if (searchData) return Promise.resolve(searchData);
    return fetch('/search.json')
      .then(function (r) { return r.json(); })
      .then(function (data) { searchData = data; return data; })
      .catch(function () { searchData = []; return searchData; });
  }

  function renderResults(query) {
    searchResults.textContent = '';
    if (query.length < 2) return;
    var q = query.toLowerCase();
    var hits = (searchData || []).filter(function (item) {
      var title = (item.title || '').toLowerCase();
      var tags = (item.tags || '').toLowerCase();
      var body = (item.content || '').toLowerCase();
      return title.includes(q) || tags.includes(q) || (q.length >= 4 && body.includes(q));
    }).slice(0, 8);

    if (!hits.length) {
      var empty = document.createElement('li');
      empty.className = 'search-results__empty';
      empty.textContent = 'No results for “' + query + '”';
      searchResults.appendChild(empty);
      return;
    }

    hits.forEach(function (item) {
      var li = document.createElement('li');
      var a = document.createElement('a');
      a.href = item.url;
      var t = document.createElement('span');
      t.className = 'search-results__title';
      t.textContent = item.title;
      var ex = document.createElement('span');
      ex.className = 'search-results__excerpt';
      ex.textContent = item.content || '';
      a.appendChild(t);
      a.appendChild(ex);
      li.appendChild(a);
      searchResults.appendChild(li);
    });
  }

  function openSearch() {
    if (!searchDialog || searchDialog.open) return;
    searchDialog.showModal();
    searchInput.value = '';
    searchResults.textContent = '';
    searchInput.focus();
    loadSearchData();
  }

  if (searchDialog && searchOpen) {
    closeOnBackdrop(searchDialog);
    searchOpen.addEventListener('click', openSearch);
    searchInput.addEventListener('input', function () {
      var query = searchInput.value.trim();
      loadSearchData().then(function () { renderResults(query); });
    });
    document.addEventListener('keydown', function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        openSearch();
      }
    });
  }

  /* BibTeX dialog */
  var bibDialog = document.getElementById('bib-dialog');
  if (bibDialog) {
    var bibPre = bibDialog.querySelector('pre');
    var copyBtn = document.getElementById('bib-copy');
    closeOnBackdrop(bibDialog);

    document.querySelectorAll('[data-bib]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        bibPre.textContent = 'Loading…';
        copyBtn.textContent = 'Copy';
        bibDialog.showModal();
        fetch(btn.getAttribute('data-bib'))
          .then(function (r) {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.text();
          })
          .then(function (text) { bibPre.textContent = text.trim(); })
          .catch(function (err) { bibPre.textContent = 'Could not load BibTeX (' + err.message + ').'; });
      });
    });

    copyBtn.addEventListener('click', function () {
      if (!navigator.clipboard) return;
      navigator.clipboard.writeText(bibPre.textContent).then(function () {
        copyBtn.textContent = 'Copied';
        setTimeout(function () { copyBtn.textContent = 'Copy'; }, 1600);
      });
    });

    var closeBtn = document.getElementById('bib-close');
    if (closeBtn) closeBtn.addEventListener('click', function () { bibDialog.close(); });
  }

  /* Back to top */
  var toTop = document.getElementById('back-to-top');
  if (toTop) {
    var onScrollTop = function () {
      toTop.classList.toggle('is-visible', window.scrollY > 600);
    };
    onScrollTop();
    window.addEventListener('scroll', onScrollTop, { passive: true });
    toTop.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  /* Article: reading progress + table of contents */
  var article = document.getElementById('post-content');
  var progress = document.getElementById('read-progress');
  if (article && progress) {
    var updateProgress = function () {
      var rect = article.getBoundingClientRect();
      var total = rect.height - window.innerHeight * 0.6;
      var ratio = total > 0 ? Math.min(Math.max(-rect.top / total, 0), 1) : 1;
      progress.style.transform = 'scaleX(' + ratio + ')';
    };
    updateProgress();
    window.addEventListener('scroll', updateProgress, { passive: true });
    window.addEventListener('resize', updateProgress);
  }

  var tocNav = document.getElementById('toc');
  if (article && tocNav) {
    var headings = Array.prototype.slice.call(article.querySelectorAll('h2, h3'));
    if (headings.length > 1) {
      var list = document.createElement('ul');
      var links = [];
      headings.forEach(function (h, i) {
        if (!h.id) h.id = 'section-' + (i + 1);
        var li = document.createElement('li');
        if (h.tagName === 'H3') li.className = 'toc__child';
        var a = document.createElement('a');
        a.href = '#' + h.id;
        a.textContent = h.textContent.replace(/^\s*\d+(\.\d+)*\.?\s+/, '');
        li.appendChild(a);
        list.appendChild(li);
        links.push(a);
      });
      tocNav.appendChild(list);
      tocNav.parentElement.classList.add('has-items');

      if ('IntersectionObserver' in window) {
        var setActive = function (id) {
          links.forEach(function (a) {
            a.classList.toggle('is-active', a.getAttribute('href') === '#' + id);
          });
        };
        var tocObserver = new IntersectionObserver(function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting) setActive(entry.target.id);
          });
        }, { rootMargin: '-90px 0px -70% 0px' });
        headings.forEach(function (h) { tocObserver.observe(h); });
      }
    }
  }

  /* Paper library filters */
  var paperSearch = document.getElementById('paper-search');
  if (paperSearch) {
    var paperStatus = document.getElementById('paper-status');
    var paperRelevance = document.getElementById('paper-relevance');
    var paperCards = Array.prototype.slice.call(document.querySelectorAll('.paper-card'));
    var paperCount = document.getElementById('paper-result-count');
    var paperEmpty = document.getElementById('paper-empty');

    var filterPapers = function () {
      var q = paperSearch.value.trim().toLowerCase();
      var visible = 0;
      paperCards.forEach(function (card) {
        var show = (!q || card.dataset.search.indexOf(q) !== -1) &&
          (!paperStatus.value || card.dataset.status === paperStatus.value) &&
          (!paperRelevance.value || card.dataset.relevance === paperRelevance.value);
        card.hidden = !show;
        if (show) visible += 1;
      });
      paperCount.textContent = visible + (visible === 1 ? ' paper' : ' papers');
      paperEmpty.hidden = visible !== 0;
    };

    paperSearch.addEventListener('input', filterPapers);
    paperStatus.addEventListener('change', filterPapers);
    paperRelevance.addEventListener('change', filterPapers);
    filterPapers();
  }

}());
