---
layout: learning-post-layout
title: "Python Style Rules"
date: 2025-12-30
lang: en
topic_url: /learning/python.html
translate_url: /cn/learning/python_guidelines/python_style_conventions.html
mathjax: false
---

## Semicolons

> 💡 **Tip**: Do not terminate your lines with semicolons, and do not use semicolons to put two statements on the same line.

## Line Length

> 💡 **Tip**: Maximum line length is 80 characters.

**Exceptions**:

1. Long import statements.
2. URLs, pathnames, or long flags in comments.
3. Long string module-level constants not containing whitespace that would be inconvenient to split across lines such as URLs or pathnames.
4. Pylint disable comments. (e.g.: `# pylint: disable=invalid-name`)

Do not use backslashes for [explicit line continuation](https://docs.python.org/3/reference/lexical_analysis.html#explicit-line-joining).

Make use of Python's [implicit line joining](https://docs.python.org/2/reference/lexical_analysis.html#implicit-line-joining) inside parentheses, brackets and braces. If necessary, you can add an extra pair of parentheses around an expression.

**✅ Correct**:

```python
foo_bar(self, width, height, color='black', design=None, x='foo',
        emphasis=None, highlight=0)

if (width == 0 and height == 0 and
    color == 'red' and emphasis == 'bold'):

(bridge_questions.clarification_on
 .average_airspeed_of.unladen_swallow) = 'African or European?'

with (
    very_long_first_expression_function() as spam,
    very_long_second_expression_function() as beans,
    third_thing() as eggs,
):
    place_order(eggs, beans, spam, beans)
```

**❌ Incorrect**:

```python
if width == 0 and height == 0 and \
    color == 'red' and emphasis == 'bold':

bridge_questions.clarification_on \
    .average_airspeed_of.unladen_swallow = 'African or European?'

with very_long_first_expression_function() as spam, \
        very_long_second_expression_function() as beans, \
        third_thing() as eggs:
    place_order(eggs, beans, spam, beans)
```

When a string literal is too long to fit on a single line, use implicit line joining with parentheses:

```python
x = ('This will build a very long long '
     'long long long long long long string')
```

Prefer to break at the highest possible syntactic level. If you must break a line more than once, try to keep the subsequent breaks at the same syntactic level.

**✅ Correct**:

```python
bridgekeeper.answer(
     name="Arthur", quest=questlib.find(owner="Arthur", perilous=True))

 answer = (a_long_line().of_chained_methods()
           .that_eventually_provides().an_answer())

 if (
     config is None
     or 'editor.language' not in config
     or config['editor.language'].use_spaces is False
 ):
   use_tabs()
```

**❌ Incorrect**:

```python
bridgekeeper.answer(name="Arthur", quest=questlib.find(
    owner="Arthur", perilous=True))

answer = a_long_line().of_chained_methods().that_eventually_provides(
    ).an_answer()

if (config is None or 'editor.language' not in config or config[
    'editor.language'].use_spaces is False):
  use_tabs()
```

Long URLs in comments can be on their own line if necessary.

**✅ Correct**:

```python
# See
# https://www.example.com/us/developer/documentation/api/content/v2.0/csv_file_name_extension_full_specification.html
```

**❌ Incorrect**:

```python
# See
# https://www.example.com/us/developer/documentation/api/content/\

# v2.0/csv_file_name_extension_full_specification.html
```

Pay attention to the indentation of the continued lines; see [Indentation](#indentation) for explanation.

Lines longer than 80 characters are allowed if [Black](https://github.com/psf/black) or [Pyink](https://github.com/google/pyink) automatic cleaners cannot shrink them. Authors are also encouraged to manually sub-divide lines following the rules above.

## Parentheses

> 💡 **Tip**: Use parentheses sparingly.

It is fine, though not required, to use parentheses around tuples. Do not use them in return statements or conditional statements unless using parentheses for implicit line continuation or to indicate a tuple.

**✅ Correct**:

```python
if foo:
    bar()
while x:
    x = bar()
if x and y:
    bar()
if not x:
    bar()
# For a one-element tuple, parentheses are more intuitive than the comma.
onesie = (foo,)
return foo
return spam, beans
return (spam, beans)
for (x, y) in dict.items(): ...
```

**❌ Incorrect**:

```python
if (x):
    bar()
if not(x):
    bar()
return (foo)
```

<div id="indentation"></div>

## Indentation

> 💡 **Tip**: Indent your code blocks with 4 spaces.

Never use tabs. For implicit line continuation, you should either align wrapped elements vertically (see [Line Length](#line_length) for examples), or use a 4-space hanging indent. The closing brace/bracket/parenthesis can either be placed at the end of the expression or on a separate line. If on a separate line, it should have the same indentation as the line that started the opening one.

**✅ Correct**:

```python
# Aligned with opening delimiter.
foo = long_function_name(var_one, var_two,
                         var_three, var_four)
meal = (spam,
        beans)

# Aligned with opening delimiter for dictionary.
foo = {
    'long_dictionary_key': value1 +
                           value2,
    ...
}

# 4-space hanging indent; nothing on first line.
foo = long_function_name(
    var_one, var_two, var_three,
    var_four)
meal = (
    spam,
    beans)

# 4-space hanging indent; nothing on first line.
# Closing delimiter on a separate line.
foo = long_function_name(
    var_one, var_two, var_three,
    var_four
)
meal = (
    spam,
    beans,
)

# 4-space hanging indent in a dictionary.
foo = {
    'long_dictionary_key':
        long_dictionary_value,
    ...
}
```

**❌ Incorrect**:

```python
# Stuff on first line forbidden.
foo = long_function_name(var_one, var_two,
    var_three, var_four)

# 2-space hanging indent forbidden.
foo = long_function_name(
  var_one, var_two, var_three,
  var_four)

# No hanging indent in dictionary.
foo = {
    'long_dictionary_key':
    long_dictionary_value,
    ...
}
```

## Trailing Commas in Sequences?

> 💡 **Tip**: Trailing commas in sequences are only recommended when the closing container token `]`, `)`, or `}` is not on the same line as the last element. The trailing comma is used by our Python auto-formatters as a hint to use one element per line.

## Shebang Line

> 💡 **Tip**: Most `.py` files do not need to start with a `#!` line. Use `#!/usr/bin/env python3` (to support virtualenv) or `#!/usr/bin/python3` according to [PEP-394](https://www.python.org/dev/peps/pep-0394/).

(Translator's note: In computer science, a [Shebang](https://en.wikipedia.org/wiki/Shebang_(Unix)) (also known as a Hashbang) is a string of characters (#!), appearing as the first two characters of a text file. If the first line of a file contains a shebang, the OS's loader parses the rest of the line as an interpreter directive and executes that program with the file path as an argument. For instance, a file beginning with #!/bin/sh would be executed by the shell script interpreter /bin/sh.)

The kernel uses this line to find the Python interpreter, but Python ignores this line when importing a module. It is only required for files that are intended to be executed directly.

## Comments and Docstrings

> 💡 **Tip**: Be sure to use the right style for module, function, and method docstrings and internal comments.

**Docstrings**

Python has a unique commenting style using docstrings. A docstring is a string that is the first statement in a package, module, class or function. Such a string can be extracted automatically using the `__doc__` attribute of the object and is used by `pydoc`. Always use the three-double-quote `"""` format for docstrings (per [PEP-257](https://www.python.org/dev/peps/pep-0257/)). A docstring should be a one-line summary (not exceeding 80 characters) ending in a period, question mark, or exclamation point. If there's more to be said, the summary must be followed by a blank line and then the rest of the text, indented the same as the start of the first line of the docstring.

**Modules**

Every file should contain a license boilerplate. Choose the appropriate boilerplate for the license used by the project (e.g., Apache 2.0, BSD, LGPL, GPL).

The file should start with a docstring describing the contents and usage of the module.

```python
"""A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of the description of the module or program
should go here, which should include an overview of the classes and
functions exported by the module and/or a usage example for the program.

Typical usage example:

foo = ClassFoo()
bar = foo.FunctionBar()
"""
```

**Test Modules**

Test files do not strictly require a module-level docstring. Only include one if it provides additional information.

For example, you can describe any special requirements for running the test, explain unusual initialization patterns, describe dependencies on external environments, etc.

```python
"""This blaze test uses golden files.

To update these files, run
`blaze run //foo/bar:foo_test -- --update_golden_files`
from your google3 folder.
"""
```

**Functions and Methods**

In this section, "function" refers to functions, methods, generators, and properties.

A function must have a docstring if it meets any of the following:

1. Part of the public API
2. Substantial in size
3. Logic is not obvious

A docstring should provide enough information for a person to call the function without having to read the function's code. The docstring should describe the function's calling syntax and its semantics, not its implementation.

Docstrings can be declarative (`"""Fetches rows from a Bigtable."""`) or imperative (`"""Fetch rows from a Bigtable."""`), but the style should be consistent within a file.

A function's features should be documented in special sections. Each section name is a heading ending in a colon.

**Args: (Arguments:)**
List all parameter names by name. Each parameter name should be followed by a colon and a space or a newline, and then a description.

**Returns: (or "Yields:" for generators)**
Describe the type and semantics of the return value. If the function only ever returns None, this section can be omitted.

**Raises: (Exceptions:)**
List all exceptions that are relevant to the interface.

```python
def fetch_smalltable_rows(
    table_handle: smalltable.Table,
    keys: Sequence[bytes | str],
    require_all_keys: bool = False,
) -> Mapping[bytes, tuple[str, ...]]:
    """Fetches rows from a Smalltable.

    Retrieves rows from a Smalltable instance for the given keys.
    If the keys are strings, they will be encoded as UTF-8.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the keys of the rows to
            fetch.  String keys will be encoded to UTF-8.
        require_all_keys: If True only rows where all keys are found will be
            returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        The keys will be byte strings.  If a key from the keys argument was not
        found in the table, that key will be missing from the dictionary
        (and require_all_keys must have been false).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """
```

**Classes**

Classes should have a docstring below the class definition describing the class. If your class has public attributes, they should be documented in an `Attributes:` section.

```python
class SampleClass(object):
    """Summary of class here.

    Attributes:
        likes_spam: A boolean indicating if we like SPAM.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, likes_spam = False):
        """Inits SampleClass with blah."""
        self.likes_spam = likes_spam
        self.eggs = 0
```

**Block and Inline Comments**

The final place to have comments is in tricky parts of the code. If you're going to have to explain it at the next code review, you should comment it now. Complicated operations should get a few lines of explanatory comments before the operation. Non-obvious code should get a comment at the end of the line.

```python
# We use a weighted dictionary search to find out where i is in
# the array.  We extrapolate an index based on the highest element
# in the array and the array length, and then binary search from
# there to find the exact index.

if i & (i-1) == 0:  # True if i is 0 or a power of two.
```

## Punctuation, Spelling, and Grammar

> 💡 **Tip**: Pay attention to punctuation, spelling, and grammar; it is easier to read a well-written comment than a poorly written one.

## Strings

> 💡 **Tip**: Use [f-strings](https://docs.python.org/zh-cn/3/reference/lexical_analysis.html#f-strings), the `%` operator, or the `format` method for formatting strings.

**✅ Correct**:

```python
x = f'name: {name}; score: {n}'
x = '%s, %s!' % (imperative, expletive)
x = '{}, {}'.format(first, second)
x = 'name: %s; score: %d' % (name, n)
x = a + b
```

**❌ Incorrect**:

```python
x = first + ', ' + second
x = 'name: ' + name + '; score: ' + str(n)
```

Avoid using the `+` and `+=` operators to accumulate a string within a loop. Instead, add each substring to a list and `''.join` the list after the loop is finished.

Conistently use either `'` or `"` for string quotes within a file. Only use the other one if needed to avoid backslashes for escaping.

Prefer `"""` for multi-line strings.

## Files, Sockets, and similar stateful resources

> 💡 **Tip**: Explicitly close files and sockets when done with them.

Prefer using the ["with" statement](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#the-with-statement) to manage files and similar resources:

```python
with open("hello.txt") as hello_file:
    for line in hello_file:
        print line
```

## TODO Comments

> 💡 **Tip**: Use TODO comments for code that is temporary, a short-term solution, or good-enough but not perfect.

TODOs should include the string `TODO` in all caps, followed by the name, e-mail address, or other identifier of the person or issue with the best context about the problem.

```python
# TODO(crbug.com/192795): Research cpufreq optimizations.
# TODO(username): Change this to use a '*' for repetitions.
```

## Import formatting

> 💡 **Tip**: Imports should be on separate lines.

Correct:

```python
from collections.abc import Mapping, Sequence
import os
import sys
from typing import Any, NewType
```

Incorrect:

```python
import os, sys
```

Imports are always put at the top of the file, just after any module comments and docstrings and before module globals and constants. Imports should be grouped with the order going from most generic to most specific:

1. `__future__` import statements.
2. Python standard library imports.
3. third-party module or package imports.
4. code repository sub-package imports.

## Statements

> 💡 **Tip**: Generally only one statement per line.

Correct:

```python
if foo: bar(foo)
```

Incorrect:

```python
if foo: bar(foo)
else:   baz(foo)
```

## Accessors (Getters and Setters)

> 💡 **Tip**: Use accessors and setters where they provide a benefit.

## Naming

> 💡 **Tip**: module_name, ClassName, method_name, ExceptionName, function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, function_parameter_name, local_var_name.

Guido's recommendations for naming:

| Type | Public | Internal |
| :--- | :--- | :--- |
| Packages | lower_with_under | |
| Modules | lower_with_under | _lower_with_under |
| Classes | CapWords | _CapWords |
| Functions | lower_with_under() | _lower_with_under() |
| Constants | CAPS_WITH_UNDER | _CAPS_WITH_UNDER |

## Main

> 💡 **Tip**: Every executable file should check `if __name__ == '__main__'` before executing the main function.

When using [absl](https://github.com/abseil/abseil-py):

```python
from absl import app

def main(argv):
    ...

if __name__ == '__main__':
    app.run(main)
```

## Function Length

> 💡 **Tip**: Prefer small and focused functions. If a function exceeds 40 lines, consider breaking it up.

## Type Annotation

**General Rules**

1. Familiarize yourself with [PEP-484](https://www.python.org/dev/peps/pep-0484/).
2. You don't need to annotate the return type for `__init__` (it must return None).
3. Annotate at least your public APIs.

**Line Breaking**

Try to follow the same indentation rules as before. If everything fits on one line, that's fine.

```python
def my_method(self, first_var: int) -> int:
    ...
```

If the return type is too long, break it and use a 4-space indent.

**NoneType**

If a variable can be None, you must declare that. Use `X | None`.

Correct:

```python
def modern_or_union(a: str | int | None, b: str | None = None) -> str:
    ...
```

**Tuples vs Lists**

```python
a: list[int] = [1, 2, 3]
b: tuple[int, ...] = (1, 2, 3)
c: tuple[int, str, float] = (1, "2", 3.5)
```

**Generics**

Always specify type parameters for generic types if possible. Otherwise, it defaults to `Any`.
