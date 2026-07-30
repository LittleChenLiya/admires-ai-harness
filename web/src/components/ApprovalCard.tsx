import { ShieldAlert } from "lucide-react";

import type { Approval } from "../types";

interface Props {
  approval: Approval;
  onDecide: (requestId: string, approved: boolean) => Promise<void>;
}

export function ApprovalCard({ approval, onDecide }: Props) {
  return (
    <div className="approval-card">
      <div className="approval-title">
        <ShieldAlert size={18} />
        <div>
          <span>{approval.risk} risk</span>
          <strong>{approval.toolName}</strong>
        </div>
      </div>
      <p>{approval.reason}</p>
      {approval.decided && (
        <p>
          The {approval.approved ? "approval" : "denial"} was saved. Retry the
          same decision to recover execution after an interruption.
        </p>
      )}
      <pre>{JSON.stringify(approval.arguments, null, 2)}</pre>
      <div className="approval-actions">
        <button
          type="button"
          className="deny"
          disabled={approval.decided && approval.approved === true}
          onClick={() => void onDecide(approval.requestId, false)}
        >
          {approval.decided && !approval.approved ? "Retry denial" : "Deny"}
        </button>
        <button
          type="button"
          className="approve"
          disabled={approval.decided && approval.approved === false}
          onClick={() => void onDecide(approval.requestId, true)}
        >
          {approval.decided && approval.approved
            ? "Resume approved action"
            : "Approve once"}
        </button>
      </div>
    </div>
  );
}
