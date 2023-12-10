# blog

My new blog, based on Quarto.

* GitHub repo (public!): https://github.com/stefanbschneider/blog
* Published blog: https://stefanbschneider.github.io/blog/
* My old blog, based on fastpages (now offline/migrated): https://github.com/stefanbschneider/old-blog

Writing/editing blog posts:

* Add/adjust new posts in the `posts/` directory. Plain markdown or `.qmd` for executable code cells.
* Use `quarto preview` to locally preview the page. There's also a VSCode extension that allows preview within VSCode (enable preview on save).
* Commit the changes to `main`. This is where the page's source lives. This does not publish the changes yet!
* To publish the changes, run `quarto publish` and confirm publishing to GitHub pages. This generates the html and pushes it to a `gh-pages` branch, from where the blog is served. No GitHub actions needed.

Further info and tips: https://quarto.org/docs/guide/
