import { useEffect, useRef, useState } from "react";

import { streamUrl } from "../api";
import type { RunEvent } from "../types";

export function useRunStream(runId: string | null, seed: RunEvent[]) {
  const [events, setEvents] = useState<RunEvent[]>(seed);
  const seen = useRef(new Set(seed.map((event) => event.event_id)));

  useEffect(() => {
    setEvents(seed);
    seen.current = new Set(seed.map((event) => event.event_id));
  }, [runId, seed]);

  useEffect(() => {
    if (!runId) return;
    const lastSequence = seed.at(-1)?.sequence ?? -1;
    const source = new EventSource(streamUrl(runId, lastSequence));
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as RunEvent;
      if (seen.current.has(event.event_id)) return;
      seen.current.add(event.event_id);
      setEvents((current) =>
        [...current, event].sort((a, b) => a.sequence - b.sequence),
      );
    };
    return () => source.close();
  }, [runId, seed]);

  return events;
}
