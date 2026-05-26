/** Stream SSE `data:` JSON payloads from a POST upload of `file`. */
export async function* streamTranscribe(
  url: string,
  file: File,
  signal?: AbortSignal,
): AsyncGenerator<unknown> {
  const form = new FormData();
  form.append("file", file, file.name);
  const resp = await fetch(url, { method: "POST", body: form, signal });
  if (!resp.ok || !resp.body) {
    throw new Error(`server returned ${resp.status}`);
  }
  const reader = resp.body
    .pipeThrough(new TextDecoderStream())
    .getReader();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let sep: number;
    while ((sep = buf.indexOf("\n\n")) !== -1) {
      const chunk = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) {
          yield JSON.parse(line.slice(6));
        }
      }
    }
  }
}
