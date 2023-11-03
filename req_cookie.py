
        #############
        import re
        import requests

        def parseCookieFile(cookiefile):
            """Parse a cookies.txt file and return a dictionary of key value pairs
            compatible with requests."""

            cookies = {}
            with open (cookiefile, 'r') as fp:
                for line in fp:
                    if not re.match(r'^\#', line):
                        lineFields = line.strip().split('\t')
                        cookies[lineFields[5]] = lineFields[6]
            return cookies

        cookies = parseCookieFile()

        response = requests.get(link, cookies=cookies)
        # print(r.content)
        #############

        # response = self.confluence.request(path=link, absolute=True)